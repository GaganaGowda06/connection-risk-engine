#!/usr/bin/env python3
# ruff: noqa: E501
"""Score the oot_week replay ledger once (test 6a). Reads the committed state of the replay and, for
the outcomes only, the oot_week tables through the guarded reader (reading_split with the
oot_replay purpose, which opens only after the guard threshold file exists). Writes the 6a table,
the 2026-10-20 gap analysis and the marker reports/serving/oot_replay_scored.json, which closes
oot_week for good.

    uv run python scripts/stage6_score.py --state data/interim/serve/state_oot
"""

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.baselines import build as BB
from rail_risk.connections import build as CB
from rail_risk.evaluation import reliability
from rail_risk.ingest.pport_extract import GATE_DONE, oot_already_scored, reading_split
from rail_risk.network_model.evaluate import paired_mae_diff
from rail_risk.serving import artifacts as AR

OUT = Path("reports/serving")
DAYS = [date(2026, 10, 17 + i) for i in range(7)]
GAP = (pd.Timestamp("2026-10-20 09:00", tz="UTC"), pd.Timestamp("2026-10-20 11:00", tz="UTC"))
SEED, N = 20261015, 1000


def ledger(state: Path):
    con = sqlite3.connect(state / "state.sqlite")
    arr = pd.read_sql_query("select * from arrival", con)
    pr = pd.read_sql_query("select * from pair", con)
    con.close()
    for df, cols in (
        (arr, ["snap_time", "pt_arr"]),
        (pr, ["snap_time", "plan_dep_out", "plan_arr_in"]),
    ):
        for c in cols:
            df[c] = pd.to_datetime(df[c], utc=True)
    for df, cols in ((arr, ["stale"]), (pr, ["stale", "nya", "held_group"])):
        for c in cols:
            df[c] = df[c].astype(bool)
    keep = {str(d) for d in DAYS}
    return arr[arr.ssd.isin(keep)].reset_index(drop=True), pr[pr.ssd.isin(keep)].reset_index(
        drop=True
    )


def labels(arr: pd.DataFrame, pr: pd.DataFrame):
    """Outcomes from the oot_week tables, read through the guarded reader."""
    root = Path("data/interim/pport")
    cfg = CB.load_config()
    mct, tc = CB.load_mct_table(), pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
    tg = []
    pl = []
    for d in DAYS:
        t = BB.day_targets(root, d).rename(columns={"tpl": "dest_tpl", "actual": "actual_arr"})
        tg.append(t.assign(ssd=str(d)))
        df, _ = CB.build_day(root, d, cfg, mct, tc, 1)  # labelled batch rows (also the parity set)
        pl.append(df)
    tg = pd.concat(tg)
    a = arr.merge(tg, on=["ssd", "rid", "dest_tpl"], how="left")
    a["gap"] = (a["actual_arr"] - a["pt_arr"]).abs()
    a = a.sort_values(
        ["ssd", "rid", "route", "horizon_min", "gap"], na_position="last"
    ).drop_duplicates(["ssd", "rid", "route", "horizon_min"])
    a.loc[a["gap"] > pd.Timedelta(hours=12), "actual_arr"] = pd.NaT
    a["y"] = (a["actual_arr"] - a["pt_arr"]).dt.total_seconds() / 60.0
    batch = pd.concat(pl)
    batch["ssd"] = batch["ssd"].astype(str)
    return a[a["y"].notna()].reset_index(drop=True), batch


def cov_ci(ssd, ok):
    d, ix = np.unique(ssd, return_inverse=True)
    n, h = np.bincount(ix).astype(float), np.bincount(ix, weights=ok)
    c = np.random.default_rng(SEED).multinomial(len(d), np.full(len(d), 1 / len(d)), size=N)
    b = (c @ h) / (c @ n)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def paired(dates, pm, pb, y):
    d = (pm - y) ** 2 - (pb - y) ** 2
    u, ix = np.unique(dates, return_inverse=True)
    n, s = np.bincount(ix).astype(float), np.bincount(ix, weights=d)
    c = np.random.default_rng(SEED).multinomial(len(u), np.full(len(u), 1 / len(u)), size=N)
    b = (c @ s) / (c @ n)
    return {
        "diff": float(s.sum() / n.sum()),
        "lo": float(np.quantile(b, 0.025)),
        "hi": float(np.quantile(b, 0.975)),
    }


def verdict(value, lo, hi, better):
    if value is None or lo is None:
        return "no test-block reference"
    if lo <= value <= hi:
        return "within"
    good = value < lo if better == "lower" else value > hi
    return "outside (better)" if good else "outside (worse)"


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="data/interim/serve/state_oot")
    ap.add_argument("--dry", action="store_true", help="rehearse on the validation guard replay")
    args = ap.parse_args()
    global DAYS, GAP
    if args.dry:
        DAYS = [date(2026, 10, 1 + i) for i in range(4)]
        GAP = (
            pd.Timestamp("2026-10-02 09:00", tz="UTC"),
            pd.Timestamp("2026-10-02 10:00", tz="UTC"),
        )
        args.state = "data/interim/serve/guard_replay/state"
    if oot_already_scored() and not args.dry:
        raise SystemExit(f"already scored: {GATE_DONE.read_text()[:200]}")
    art = AR.load()
    held = set(art["held"])
    arr, pr = ledger(Path(args.state))
    import contextlib

    with contextlib.nullcontext() if args.dry else reading_split("oot_week", "oot_replay"):
        a, batch = labels(arr, pr)
    t4 = json.loads(Path("reports/network_model/stage4/test_results.json").read_text())
    t4b = json.loads(Path("reports/network_model/stage4b/second_scoring_results.json").read_text())[
        "test_primary_no_history"
    ]
    t5 = json.loads(Path("reports/connections/stage5/test_results.json").read_text())["cells"]
    rows = []

    def add(name, value, ref_lo, ref_hi, better, note=""):
        rows.append(
            {
                "metric": name,
                "replay": value,
                "test_lo": ref_lo,
                "test_hi": ref_hi,
                "verdict": verdict(value, ref_lo, ref_hi, better),
                "note": note,
            }
        )

    out: dict = {
        "arrival_rows_scored": len(a),
        "held_out_arrival_rows": int(a.route.isin(held).sum()),
    }
    h = a[a.route.isin(held)].reset_index(drop=True)
    y = h.y.to_numpy()
    # test 2: paired MAE
    ref = {(r["bar"], r["baseline"], r["horizon_min"]): r for r in t4["paired_mae"]}
    for bname, col, bar in (
        ("persistence", "pers_point", "cold"),
        ("historical", "hist_point", "history"),
    ):
        for hz in (15, 60):
            m = (h.horizon_min == hz).to_numpy()
            r = paired_mae_diff(
                h.point.to_numpy()[m] - y[m],
                h[col].to_numpy()[m] - y[m],
                h.ssd.to_numpy()[m],
                N,
                SEED,
            )
            rr = ref[(bar, bname, hz)]
            add(
                f"test2 MAE diff vs {bname} at {hz} (model minus baseline, minutes)",
                r["diff"],
                rr["lo"],
                rr["hi"],
                "lower",
                f"replay CI [{r['lo']:.3f}, {r['hi']:.3f}], n {int(m.sum())}",
            )
    # test 3: coverage
    ok = ((h.final_lo <= h.y) & (h.y <= h.final_hi)).astype(float).to_numpy()
    add(
        "test3 coverage overall (held-out routes)",
        float(ok.mean()),
        *t4b["adaptive"]["overall_ci"],
        "target",
        f"replay CI {np.round(cov_ci(h.ssd.to_numpy(), ok), 4).tolist()}, n {len(h)}",
    )
    for hz in (15, 60):
        m = (h.horizon_min == hz).to_numpy()
        add(
            f"test3 coverage at {hz}",
            float(ok[m].mean()),
            *t4b["adaptive"][str(hz)]["ci"],
            "target",
            f"n {int(m.sum())}",
        )
    per = pd.Series(ok).groupby(h.route.to_numpy()).agg(["mean", "size"])
    out["coverage_per_route"] = {
        k: {"coverage": float(v["mean"]), "n": int(v["size"])} for k, v in per.iterrows()
    }
    out["lowest_route"] = {
        "route": per["mean"].idxmin(),
        "coverage": float(per["mean"].min()),
        "test3_rule_min_0.70_met": bool(per["mean"].min() >= 0.70),
    }
    out["coverage_unguarded_aci"] = float(((h.aci_lo <= h.y) & (h.y <= h.aci_hi)).mean())
    # test 4: P(delay >= 15)
    for hz in (15, 60):
        m = (h.horizon_min == hz).to_numpy()
        o = (y[m] >= 15).astype(float)
        r = reliability(
            h.ssd.to_numpy()[m], h.p_late_final.to_numpy()[m], o, bins=10, resamples=N, seed=SEED
        )
        t = t4["p_late"][f"model_{hz}"]
        add(
            f"test4 Brier P(delay>=15) at {hz}",
            r["brier"]["value"],
            t["brier_lo"],
            t["brier_hi"],
            "lower",
            f"replay CI [{r['brier']['ci_low']:.4f}, {r['brier']['ci_high']:.4f}], base rate {o.mean():.4f}",
        )
        add(
            f"test4 ECE P(delay>=15) at {hz}",
            r["ece"]["value"],
            t["ece_lo"],
            t["ece_hi"],
            "lower",
            f"replay CI [{r['ece']['ci_low']:.4f}, {r['ece']['ci_high']:.4f}]",
        )
        pd_ = paired(h.ssd.to_numpy()[m], h.p_late_final.to_numpy()[m], h.hist_p.to_numpy()[m], o)
        tr = t4["brier_paired"][f"vs_historical_history_{hz}"]
        add(
            f"test4 Brier diff vs historical at {hz} (model minus historical)",
            pd_["diff"],
            tr["lo"],
            tr["hi"],
            "lower",
            f"replay CI [{pd_['lo']:.4f}, {pd_['hi']:.4f}]",
        )
    # test 5: P(miss), held-out groups
    lab = batch[(batch.out_cancel == 0) & batch.miss.notna()][
        ["ssd", "hub", "in_rid", "out_rid", "horizon_min", "miss"]
    ]
    p = pr.merge(lab, on=["ssd", "hub", "in_rid", "out_rid", "horizon_min"], how="inner")
    hp = p[p.held_group].reset_index(drop=True)
    out["pair_rows"] = {
        "ledger": len(pr),
        "labelled": len(p),
        "held_out_groups": len(hp),
        "held_out_not_arrived": int(hp.nya.sum()),
    }
    for hub in ("BHM", "MAN", "BRI"):
        s = hp[(hp.hub == hub) & hp.nya]
        ya = s.miss.to_numpy(float)
        pdiff = paired(s.ssd.to_numpy(), s.p_final.to_numpy(), s.p_bar.to_numpy(), ya)
        ref = t5[f"held|not_arrived|{hub}|all"]
        add(
            f"test5 {hub} Brier diff vs gap-bucket, not-yet-arrived (final minus bar)",
            pdiff["diff"],
            ref["paired_brier"]["lo"],
            ref["paired_brier"]["hi"],
            "lower",
            f"replay CI [{pdiff['lo']:.4f}, {pdiff['hi']:.4f}], n {len(s)}",
        )
        r = reliability(s.ssd.to_numpy(), s.p_final.to_numpy(), ya, bins=10, resamples=N, seed=SEED)
        add(
            f"test5 {hub} ECE, not-yet-arrived",
            r["ece"]["value"],
            ref["ece"]["ci_low"],
            ref["ece"]["ci_high"],
            "lower",
            f"replay CI [{r['ece']['ci_low']:.4f}, {r['ece']['ci_high']:.4f}]",
        )
        out.setdefault("pmiss_vs_logistic", {})[hub] = {
            "brier_final": brier(s.p_final, ya), "brier_gap_bucket": brier(s.p_bar, ya), "brier_logistic_c": brier(s.p_logit, ya),
            "diff_final_minus_logistic": paired(s.ssd.to_numpy(), s.p_final.to_numpy(), s.p_logit.to_numpy(), ya),
            "note": "logistic (c) was not scored on the test block, so there is no test-block reference for this line"}  # fmt: skip
    # parity of the service with the batch pipeline on pairs that were not stale
    from rail_risk.serving import engine as E  # noqa: F401

    model = __import__("joblib").load(AR.MODELS5 / f"{AR.PMISS_RUN}.joblib")
    from rail_risk.connections import data as DD

    bf = DD.flag_groups(batch, CB.load_config()).reset_index(drop=True)
    pb = model.predict(bf)["lgbm_cal"].to_numpy()
    bf["p_batch"] = pb
    m = pr.merge(
        bf[["ssd", "hub", "in_rid", "out_rid", "horizon_min", "p_batch"]],
        on=["ssd", "hub", "in_rid", "out_rid", "horizon_min"],
        how="inner",
    )
    from rail_risk.serving.replay_gaps import GAPS as VAL_GAPS

    windows = [(GAP[0], GAP[1])] if not args.dry else VAL_GAPS
    inside = pd.Series(False, index=m.index)
    for g0, g1 in windows:
        inside |= (m.snap_time >= g0) & (m.snap_time < g1 + pd.Timedelta(minutes=240))
    out["parity_rows_in_affected_windows_excluded"] = int(inside.sum())
    m = m[~m.stale & ~inside]
    out["parity_pairs_not_stale"] = {
        "rows": len(m),
        "max_abs_diff": float((m.p_b - m.p_batch).abs().max()),
        "rows_over_1e-6": int(((m.p_b - m.p_batch).abs() > 1e-6).sum()),
    }
    # the 2026-10-20 gap
    lo, hi = GAP[0], GAP[1] + pd.Timedelta(minutes=240)
    g = p[(p.snap_time >= lo) & (p.snap_time < hi)]
    ga = a[(a.snap_time >= lo) & (a.snap_time < hi)]
    gh = ga[ga.route.isin(held)]
    gap = {"window_utc": [str(lo), str(hi)], "pair_rows": len(g), "pair_stale_share": float(g.stale.mean()) if len(g) else None,
           "pair_brier_with_guard": brier(g.p_final, g.miss) if len(g) else None, "pair_brier_without_guard": brier(g.p_b, g.miss) if len(g) else None, "pair_brier_gap_bucket": brier(g.p_bar, g.miss) if len(g) else None,
           "arrival_rows": len(ga), "arrival_held_out_rows": len(gh), "arrival_stale_share": float(ga.stale.mean()) if len(ga) else None}  # fmt: skip
    if len(gh):
        gy = gh.y
        gap["held_out_coverage_with_guard"] = float(
            ((gh.final_lo <= gy) & (gy <= gh.final_hi)).mean()
        )
        gap["held_out_coverage_without_guard"] = float(
            ((gh.aci_lo <= gy) & (gy <= gh.aci_hi)).mean()
        )
        o = (gy >= 15).astype(float)
        gap["held_out_brier_p_late_with_guard"], gap["held_out_brier_p_late_without_guard"] = (
            brier(gh.p_late_final, o),
            brier(gh.p_late, o),
        )
    gp = g[g.stale]
    gap["pair_stale_only"] = {
        "rows": len(gp),
        "brier_with_guard": brier(gp.p_final, gp.miss) if len(gp) else None,
        "brier_without_guard": brier(gp.p_b, gp.miss) if len(gp) else None,
    }
    out["gap_2026_10_20"] = gap
    out["table_6a"] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "validation_dry" if args.dry else "oot"
    (OUT / f"{tag}_results.json").write_text(json.dumps(out, indent=1, default=float))
    ld = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
    md = ["| metric | replay | test-block 95% CI | verdict |", "|---|---|---|---|"] + [
        f"| {r['metric']} | {ld(r['replay'])} | [{ld(r['test_lo'])}, {ld(r['test_hi'])}] | {r['verdict']} |"
        for r in rows
    ]
    (OUT / f"{tag}_table_6a.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "pair_rows",
                    "parity_pairs_not_stale",
                    "gap_2026_10_20",
                    "lowest_route",
                    "pmiss_vs_logistic",
                )
            },
            indent=1,
            default=float,
        )
    )
    if args.dry:
        return  # a rehearsal never writes the marker
    GATE_DONE.write_text(
        json.dumps({"oot_week_scored": True, "state": args.state, "arrival_rows": len(a),
                    "pair_rows": len(p)}) + "\n"
    )  # fmt: skip


if __name__ == "__main__":
    main()
