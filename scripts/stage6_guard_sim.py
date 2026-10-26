#!/usr/bin/env python3
# ruff: noqa: E501
"""Compare three guard designs on the simulated gaps (scripts/stage6_gap_sim.py ledgers).

Train_pool days only (validation 2026-09-25 to 10-08 and train 2026-09-19 to 09-22); never test,
February or oot_week. Rows are the held-out routes (arrival) and held-out service groups (pairs),
the same units the tests score. Days alternate between a FIT half and a SCORE half (the halves are in
sim_gaps.json, fixed before any replay): every parameter of every design is fitted on the fit half
and all numbers that decide are from the score half.

Rule for the choice, written before the scoring half was looked at: among the designs whose arrival
coverage on affected score-half rows is within [0.75, 0.85] and whose affected-row P(delay >= 15)
Brier is no more than 5% worse than keeping the model, take the lowest P(miss) Brier on all
score-half pair rows (affected and intact together); if none qualifies, take widen.

    uv run python scripts/stage6_guard_sim.py [--reuse-labels]
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from stage6_gap_sim import BLOCKS, WORK  # noqa: I001
from stage6_gap_sim import OUT as GAPS

from rail_risk.baselines import build as BB
from rail_risk.connections import build as CB
from rail_risk.evaluation import _bin_stats
from rail_risk.serving import artifacts as AR
from rail_risk.serving.guard import Guard

CACHE = Path("data/interim/serve/gapsim")
OUTDIR = Path("reports/serving")
RECOVERY = pd.Timedelta(minutes=240)
SEED = 20261026


def load_ledger(block: str):
    import sqlite3

    con = sqlite3.connect(WORK / block / "state" / "state.sqlite")
    arr = pd.read_sql_query("select * from arrival", con)
    pr = pd.read_sql_query("select * from pair", con)
    con.close()
    for df, cols in (
        (arr, ["snap_time", "pt_arr"]),
        (pr, ["snap_time", "plan_dep_out", "plan_arr_in"]),
    ):
        for c in cols:
            df[c] = pd.to_datetime(df[c], utc=True)
    d0, d1, _ = BLOCKS[block]
    keep = {str(d0 + timedelta(days=i)) for i in range((d1 - d0).days + 1)}
    return arr[arr.ssd.isin(keep)].assign(block=block), pr[pr.ssd.isin(keep)].assign(block=block)


def label_block(block: str, arr: pd.DataFrame, pr: pd.DataFrame):
    root = Path("data/interim/pport")
    cfg = CB.load_config()
    mct, tc = CB.load_mct_table(), pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
    d0, d1, _ = BLOCKS[block]
    tg, pl = [], []
    for i in range((d1 - d0).days + 1):
        d = d0 + timedelta(days=i)
        t = BB.day_targets(root, d).rename(columns={"tpl": "dest_tpl", "actual": "actual_arr"})
        tg.append(t.assign(ssd=str(d)))
        df, _ = CB.build_day(root, d, cfg, mct, tc, 1)
        pl.append(df[["ssd", "hub", "in_rid", "out_rid", "horizon_min", "miss", "out_cancel"]])
    tg = pd.concat(tg)
    a = arr.merge(tg, on=["ssd", "rid", "dest_tpl"], how="left")
    a["gap"] = (a["actual_arr"] - a["pt_arr"]).abs()
    a = a.sort_values(
        ["ssd", "rid", "route", "horizon_min", "gap"], na_position="last"
    ).drop_duplicates(["ssd", "rid", "route", "horizon_min"])
    a.loc[a["gap"] > pd.Timedelta(hours=12), "actual_arr"] = pd.NaT
    a["y"] = (a["actual_arr"] - a["pt_arr"]).dt.total_seconds() / 60.0
    lab = pd.concat(pl)
    lab["ssd"] = lab["ssd"].astype(str)
    p = pr.merge(lab, on=["ssd", "hub", "in_rid", "out_rid", "horizon_min"], how="inner")
    p = p[(p.out_cancel == 0) & p.miss.notna()]
    return a[a["y"].notna()].drop(columns=["actual_arr", "gap"]), p


def load_all(reuse: bool):
    fa, fp = CACHE / "labelled_arrival.parquet", CACHE / "labelled_pair.parquet"
    if reuse and fa.exists() and fp.exists():
        return pd.read_parquet(fa), pd.read_parquet(fp)
    A, P = [], []
    for block in BLOCKS:
        arr, pr = load_ledger(block)
        a, p = label_block(block, arr, pr)
        A.append(a)
        P.append(p)
    A, P = pd.concat(A, ignore_index=True), pd.concat(P, ignore_index=True)
    for df in (A, P):
        df["stale"] = df["stale"].astype(bool)
    A.to_parquet(fa)
    P.to_parquet(fp)
    return A, P


def tag(df: pd.DataFrame, gaps: list[dict]) -> pd.DataFrame:
    """affected (T inside a removed window or the 240 minutes after it), gap length and half."""
    df = df.copy()
    df["affected"], df["gap_min"] = False, 0
    for g in gaps:
        a, b = pd.Timestamp(g["start"]), pd.Timestamp(g["end"])
        m = (df.snap_time >= a) & (df.snap_time < b + RECOVERY)
        df.loc[m, "affected"] = True
        df.loc[m, "gap_min"] = g["minutes"]
    half = {}
    for d0, d1, _ in BLOCKS.values():
        for i in range((d1 - d0).days + 1):
            half[str(d0 + timedelta(days=i))] = "fit" if i % 2 == 0 else "score"
    df["half"] = df["ssd"].map(half)
    return df


def interval_score(y, lo, hi, alpha=0.2):
    w = hi - lo
    return float(np.mean(w + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))))


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(p, y):
    return float(_bin_stats(np.asarray(p), np.asarray(y, dtype=float), 10)[1])


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _gate_u(df, flag_min, window):
    s60, s240 = (
        np.maximum(df.feed_age_min, df.feed_gap60).to_numpy(),
        np.maximum(df.feed_age_min, df.feed_gap240).to_numpy(),
    )
    gate = (np.maximum(df.feed_age_min, df[f"feed_gap{window}"]) > flag_min).to_numpy()
    return (
        np.log(np.maximum(s60, flag_min) / flag_min),
        np.log(np.maximum(s240, flag_min) / flag_min),
        gate,
    )


def fit_blend(df, p_model, p_fb, y, with_diff=True, l2=0.02, flag_min=12.0, window=240):
    """Weight w = gate * sigmoid(a + b1 u1 + b2 u2 [+ c d]) on the fallback, by Brier on the fit rows;
    gate is zero while the feed is fresh."""
    u1, u2, gate = _gate_u(df, flag_min, window)
    d = p_fb - p_model
    n = 4 if with_diff else 3

    def f(th):
        z = th[0] + th[1] * u1 + th[2] * u2 + (th[3] * d if with_diff else 0)
        w = gate / (1 + np.exp(-np.clip(z, -30, 30)))
        p = (1 - w) * p_model + w * p_fb
        return np.mean((p - y) ** 2) + l2 * np.sum(np.square(th[1:])) * 1e-3

    best = None
    for a0 in (-6.0, -3.0, -1.0):
        r = minimize(
            f,
            np.array([a0] + [0.0] * (n - 1)),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-9},
        )
        if best is None or r.fun < best.fun:
            best = r
    return [float(x) for x in best.x]


def fit_range_blend(df, y, l2=0.02, flag_min=12.0, window=240):
    u1, u2, gate = _gate_u(df, flag_min, window)

    def f(th):
        w = gate / (1 + np.exp(-np.clip(th[0] + th[1] * u1 + th[2] * u2, -30, 30)))
        lo = (1 - w) * df.aci_lo + w * df.static_lo
        hi = (1 - w) * df.aci_hi + w * df.static_hi
        return interval_score(y, lo.to_numpy(), hi.to_numpy()) + l2 * np.sum(np.square(th[1:]))

    best = None
    for a0 in (-6.0, -3.0, -1.0):
        r = minimize(
            f,
            np.array([a0, 0.0, 0.0]),
            method="Nelder-Mead",
            options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-7},
        )
        if best is None or r.fun < best.fun:
            best = r
    return [float(x) for x in best.x]


def fit_designs(A, P):
    """Parameters of the three designs from the fit half only."""
    fa, fp = A[A.half == "fit"], P[P.half == "fit"]
    out = {}
    # (b) binary: threshold and window by pooled pair Brier with at most 0.5% false-stale on intact rows
    grid = []
    for th in (3, 5, 8, 12, 20, 30, 45, 60):
        for R in (60, 120, 240):
            s = np.maximum(fp.feed_age_min, fp[f"feed_gap{R}"]) > th
            false = float(s[~fp.affected].mean())
            p = np.where(s, fp.p_bar, fp.p_b)
            grid.append({"th": th, "R": R, "false": false, "brier": brier(p, fp.miss)})
    ok = [g for g in grid if g["false"] <= 0.005] or grid
    b = min(ok, key=lambda g: g["brier"])
    out["binary_grid"] = grid
    out["binary"] = Guard(
        design="binary", bin_min=b["th"], bin_window=b["R"], flag_min=b["th"], flag_window=b["R"]
    )
    # (a) widen: flag as before, k and window by mean interval score on all fit arrival rows
    wg = []
    for R in (60, 120, 240):
        for k in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            g = Guard(design="widen", widen_k=k, widen_window=R, flag_window=R)
            r = g.arrivals(fa)
            wg.append(
                {
                    "R": R,
                    "k": k,
                    "iscore": interval_score(
                        fa.y.to_numpy(), r.final_lo.to_numpy(), r.final_hi.to_numpy()
                    ),
                }
            )
    wb = min(wg, key=lambda g: g["iscore"])
    out["widen_grid"] = wg
    out["widen"] = Guard(design="widen", widen_k=wb["k"], widen_window=wb["R"], flag_window=wb["R"])
    # (c) blend
    y15 = (fa.y >= 15).astype(float).to_numpy()
    pair_w = fit_blend(fp, fp.p_b.to_numpy(), fp.p_bar.to_numpy(), fp.miss.to_numpy().astype(float))
    late_w = fit_blend(fa, fa.p_late.to_numpy(), fa.hist_p.to_numpy(), y15)
    range_w = fit_range_blend(fa, fa.y.to_numpy())
    out["blend"] = Guard(
        design="blend", flag_window=240, pair_w=pair_w, late_w=late_w, range_w=range_w
    )
    out["none"] = Guard(design="none")
    return out


def evaluate(g: Guard, A, P, label: str):
    res = {}
    for part in ("score", "fit"):
        a, p = A[A.half == part].reset_index(drop=True), P[P.half == part].reset_index(drop=True)
        ra, rp = g.arrivals(a), g.pairs(p)
        y15 = (a.y >= 15).astype(float).to_numpy()
        cell = {}
        for nm, ma, mp in (
            ("affected", a.affected.to_numpy(), p.affected.to_numpy()),
            ("intact", ~a.affected.to_numpy(), ~p.affected.to_numpy()),
            ("all", np.ones(len(a), bool), np.ones(len(p), bool)),
        ):
            nya = mp & p.nya.to_numpy()
            ya = a.y.to_numpy()[ma]
            lo, hi = ra.final_lo.to_numpy()[ma], ra.final_hi.to_numpy()[ma]
            cell[nm] = {
                "pairs": int(mp.sum()),
                "pmiss_brier": brier(rp.p_final.to_numpy()[mp], p.miss.to_numpy()[mp]),
                "pmiss_ece_nya": ece(rp.p_final.to_numpy()[nya], p.miss.to_numpy()[nya])
                if nya.sum() > 50
                else None,
                "arrivals": int(ma.sum()),
                "coverage": float(((lo <= ya) & (ya <= hi)).mean()),
                "mean_width": float((hi - lo).mean()),
                "interval_score": interval_score(ya, lo, hi),
                "plate_brier": brier(ra.p_late_final.to_numpy()[ma], y15[ma]),
            }
        cell["stale_share_pairs_affected"] = float(rp.stale[p.affected].mean())
        cell["stale_share_pairs_intact"] = float(rp.stale[~p.affected].mean())
        cell["mean_fallback_w_affected"] = float(rp.fallback_w[p.affected].mean())
        cell["mean_fallback_w_intact"] = float(rp.fallback_w[~p.affected].mean())
        res[part] = cell
    return res


def by_gap_length(g: Guard, A, P):
    """Score-half P(miss) Brier on affected pair rows per gap length, design against keeping the model."""
    p = P[(P.half == "score") & P.affected].reset_index(drop=True)
    rp, base = g.pairs(p), Guard(design="none").pairs(p)
    out = {}
    for m, ix in p.groupby("gap_min").indices.items():
        out[int(m)] = {
            "rows": len(ix),
            "brier": brier(rp.p_final.to_numpy()[ix], p.miss.to_numpy()[ix]),
            "brier_model": brier(base.p_final.to_numpy()[ix], p.miss.to_numpy()[ix]),
        }
    return out


def paired_ci(gA, gB, P, resamples=1000):
    """Score-half pair Brier difference (A minus B) over all rows, bootstrap over days."""
    p = P[P.half == "score"].reset_index(drop=True)
    d = (gA.pairs(p).p_final.to_numpy() - p.miss.to_numpy()) ** 2 - (
        gB.pairs(p).p_final.to_numpy() - p.miss.to_numpy()
    ) ** 2
    u, ix = np.unique(p.ssd.to_numpy(), return_inverse=True)
    n, s = np.bincount(ix).astype(float), np.bincount(ix, weights=d)
    c = np.random.default_rng(SEED).multinomial(len(u), np.full(len(u), 1 / len(u)), size=resamples)
    b = (c @ s) / (c @ n)
    return {
        "diff": float(s.sum() / n.sum()),
        "lo": float(np.quantile(b, 0.025)),
        "hi": float(np.quantile(b, 0.975)),
        "days": int(len(u)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-labels", action="store_true")
    ap.add_argument(
        "--write", action="store_true", help="write the choice to reports/serving/guard_v2.json"
    )
    args = ap.parse_args()
    gaps = json.loads(GAPS.read_text())["gaps"]
    A, P = load_all(args.reuse_labels)
    held = set(AR.load()["held"])
    A = A[A.route.isin(held)]
    P = P[P.held_group.astype(bool)]
    A, P = tag(A, gaps).reset_index(drop=True), tag(P, gaps).reset_index(drop=True)
    P["nya"] = P["nya"].astype(bool)
    print("rows", len(A), len(P), "affected", int(A.affected.sum()), int(P.affected.sum()))
    designs = fit_designs(A, P)
    names = ["none", "widen", "binary", "blend"]
    res = {n: evaluate(designs[n], A, P, n) for n in names}
    # the current Stage 6 guard (v1: silence over 12 min in the last 60) as a reference line
    v1 = Guard(design="binary", bin_min=12, bin_window=60, flag_min=12, flag_window=60)
    res["v1_current"] = evaluate(v1, A, P, "v1")
    # the rule
    ok = []
    for n in ("widen", "binary", "blend"):
        sc = res[n]["score"]
        cov_ok = 0.75 <= sc["affected"]["coverage"] <= 0.85
        late_ok = (
            sc["affected"]["plate_brier"] <= 1.05 * res["none"]["score"]["affected"]["plate_brier"]
        )
        if cov_ok and late_ok:
            ok.append(n)
    pick = min(ok, key=lambda n: res[n]["score"]["all"]["pmiss_brier"]) if ok else "widen"
    out = {
        "rows": {
            "arrival_heldout": len(A),
            "pair_heldout": len(P),
            "affected_pairs": int(P.affected.sum()),
        },
        "gaps": len(gaps),
        "fit_gaps": sum(g["half"] == "fit" for g in gaps),
        "score_gaps": sum(g["half"] == "score" for g in gaps),
        "params": {n: designs[n].to_json() for n in names[1:]},
        "binary_grid": designs["binary_grid"],
        "widen_grid": designs["widen_grid"],
        "results": res,
        "qualifying": ok,
        "pick": pick,
        "paired_vs_none": {
            n: paired_ci(designs[n], designs["none"], P) for n in ("widen", "binary", "blend")
        },
        "by_gap_length_score_half": {
            n: by_gap_length(designs[n], A, P) for n in ("none", "binary", "blend")
        },
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "guard_sim.json").write_text(json.dumps(out, indent=1, default=float))
    print(
        json.dumps(
            {k: out[k] for k in ("rows", "qualifying", "pick", "paired_vs_none")},
            indent=1,
            default=float,
        )
    )
    for n in ["none", "v1_current", "widen", "binary", "blend"]:
        s = res[n]["score"]
        print(
            f"{n:11s} pmiss all {s['all']['pmiss_brier']:.4f} aff {s['affected']['pmiss_brier']:.4f} ece_nya aff {s['affected']['pmiss_ece_nya']} | cov aff {s['affected']['coverage']:.3f} all {s['all']['coverage']:.3f} | plate aff {s['affected']['plate_brier']:.4f} all {s['all']['plate_brier']:.4f}"
        )
    if args.write:
        d = designs[pick].to_json()
        d.update(
            {
                "chosen_on": "simulated gaps, scripts/stage6_guard_sim.py, fit half; scored on the other half; train and validation days only"
            }
        )
        (OUTDIR / "guard_v2.json").write_text(json.dumps(d, indent=1))
        print("wrote guard_v2.json", pick)


if __name__ == "__main__":
    main()
