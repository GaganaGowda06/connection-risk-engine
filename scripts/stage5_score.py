#!/usr/bin/env python3
"""Stage 5 scoring of a frozen run: the test block once, then February once as its own line.

    uv run python scripts/stage5_score.py --run-id RUN --block test
    uv run python scripts/stage5_score.py --run-id RUN --block february

Primary set: pairs whose inbound or outbound service group was held out (never in a fit, tuning
fold or calibration). Cells per hub and horizon: Brier of the model and of the gap-bucket bar with
a paired date-bootstrap CI on the difference (model minus bar, below zero is better), and ECE with a
date-bootstrap CI on pairs whose inbound has not yet arrived by T. Leave-hub-out: the model trained
on the other two hubs scored on all pairs of the third, against a bar fitted on those two hubs.
The four originally configured pairs are scored on their own. Each block refuses to be scored twice.
Test 5 verdict rule (written before the test run): a hub passes when, on held-out-group pairs,
the paired Brier CI is below zero pooled over horizons for all pairs and for not-yet-arrived pairs,
and ECE on not-yet-arrived pairs is at most 0.05 pooled and at every horizon; test 5 needs 2 hubs.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from rail_risk.connections import baseline as BL
from rail_risk.connections import build as B
from rail_risk.connections import data as DD
from rail_risk.connections import online as O
from rail_risk.evaluation import reliability
from rail_risk.network_model.run import git_state

OUT = Path("reports/connections/stage5")
MODELS = Path("data/interim/stage5/models")
HORIZONS = (5, 15, 30, 60)


def paired(dates, pm, pb, y, resamples, seed):
    d = (pm - y) ** 2 - (pb - y) ** 2
    u, ix = np.unique(dates, return_inverse=True)
    n, s = np.bincount(ix).astype(float), np.bincount(ix, weights=d)
    counts = np.random.default_rng(seed).multinomial(
        len(u), np.full(len(u), 1 / len(u)), size=resamples
    )
    boot = (counts @ s) / (counts @ n)
    return {"diff": float(s.sum() / n.sum()), "lo": float(np.quantile(boot, 0.025)),
            "hi": float(np.quantile(boot, 0.975))}  # fmt: skip


def cell(df, pm, pb, resamples, seed, ece=True):
    y = df["miss"].to_numpy(float)
    dates = df["ssd"].to_numpy()
    out = {"n": int(len(df)), "base_rate": float(y.mean()),
           "brier_model": float(np.mean((pm - y) ** 2)),
           "brier_gap_bucket": float(np.mean((pb - y) ** 2)),
           "paired_brier": paired(dates, pm, pb, y, resamples, seed)}  # fmt: skip
    if ece:
        r = reliability(dates, pm, y, bins=10, resamples=resamples, seed=seed)
        out["ece"] = r["ece"]
    return out


def table(df, pm, pb, cfg, subset, group_label, resamples, seed):
    res = {}
    for hub in ("ALL", *cfg["hubs"]):
        for hz in (*HORIZONS, "all"):
            m = np.ones(len(df), bool)
            if hub != "ALL":
                m &= (df["hub"] == hub).to_numpy()
            if hz != "all":
                m &= (df["horizon_min"] == hz).to_numpy()
            if subset == "not_arrived":
                m &= df["nya"].to_numpy()
            if m.sum() < 30:
                res[f"{group_label}|{subset}|{hub}|{hz}"] = {
                    "n": int(m.sum()),
                    "skipped": "fewer than 30",
                }
                continue
            res[f"{group_label}|{subset}|{hub}|{hz}"] = cell(
                df[m], pm[m], pb[m], resamples, seed, ece=(subset == "not_arrived")
            )
    return res


def score_days(models, pick, days, block, cfg):
    """Predict day by day (memory), keeping only slim columns."""
    parts = []
    for d in days:
        df = DD.load_block(block, [d])
        df = DD.labelled(DD.flag_groups(df, cfg)).reset_index(drop=True)
        if df.empty:
            continue
        slim = df[["ssd", "hub", "in_rid", "out_rid", "horizon_min", "held_group", "slack_min",
                   "miss", "snap_time", "act_dep_out", "in_arr_late_s"]].copy()  # fmt: skip
        slim["nya"] = slim.pop("in_arr_late_s").isna()
        for name, m in models.items():
            slim[f"p_{name}"] = m.predict(df)[f"{pick}_cal"].to_numpy()
        parts.append(slim)
    return pd.concat(parts, ignore_index=True)


def verdict(res, cfg):
    out = {}
    for hub in cfg["hubs"]:
        a = res[f"held|all_pairs|{hub}|all"]["paired_brier"]["hi"] < 0
        n = res[f"held|not_arrived|{hub}|all"]["paired_brier"]["hi"] < 0
        eces = [
            res[f"held|not_arrived|{hub}|{h}"].get("ece", {}).get("value")
            for h in (*HORIZONS, "all")
        ]
        e = all(v is not None and v <= 0.05 for v in eces)
        out[hub] = {"brier_ci_below_zero_all": bool(a), "brier_ci_below_zero_not_arrived": bool(n),
                    "ece_ok": bool(e), "pass": bool(a and n and e)}  # fmt: skip
    out["hubs_passing"] = sum(v["pass"] for v in out.values() if isinstance(v, dict))
    out["test5_met"] = out["hubs_passing"] >= 2
    return out


def original_pairs(df, block):
    """Flag the pairs of the four configured interchanges (Stage 2 definition by service)."""
    if block == "february":
        from rail_risk.baselines.pairs import build_pairs_day
        from rail_risk.clean.run import route_specs  # noqa: F401
        from rail_risk.config import load_routes
        from rail_risk.ingest.pport_extract import reading_split

        routes = load_routes("config/routes.yaml")
        tc = pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
        frames = []
        with reading_split("winter_test", "february_stress"):
            for d in sorted(df["ssd"].unique()):
                p = build_pairs_day(Path("data/interim/pport"), date.fromisoformat(d), routes, tc)
                if len(p):
                    frames.append(p[["ssd", "hub", "outbound", "in_rid", "out_rid"]])
        pairs = pd.concat(frames)
    else:
        root = Path("data/interim/stage2/pairs")
        files = [
            f
            for d in sorted(df["ssd"].unique())
            for f in (root / f"service_date={d}").glob("*.parquet")
        ]
        if not files:  # a dry run on days Stage 2 never built pairs for
            return df.iloc[:0].assign(outbound=pd.Series(dtype=str))
        pairs = pd.concat(
            [
                pd.read_parquet(f, columns=["ssd", "hub", "outbound", "in_rid", "out_rid"])
                for f in files
            ]
        )
    pairs["ssd"] = pairs["ssd"].astype(str)
    return df.merge(
        pairs.drop_duplicates(["ssd", "hub", "in_rid", "out_rid"]),
        on=["ssd", "hub", "in_rid", "out_rid"],
        how="inner",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--block", choices=("test", "february", "validation"), required=True)
    args = ap.parse_args()
    marker = OUT / f"{args.block}_scored.json"
    dry = args.block == "validation"  # a debugging dry run: no marker, no MLflow
    if marker.exists() and not dry:
        raise SystemExit(f"{args.block} was already scored: {marker.read_text()[:200]}")
    head, dirty = git_state()
    cfg = B.load_config()
    fit = json.loads((OUT / "final_fit.json").read_text())
    if fit["run_id"] != args.run_id:
        raise SystemExit("run id does not match reports/connections/stage5/final_fit.json")
    pick, eta = fit["option"], fit["online_eta"]
    res_n, seed = cfg["bootstrap"]["resamples"], cfg["bootstrap"]["seed"]
    models = {"main": joblib.load(MODELS / f"{args.run_id}.joblib")}
    for h in cfg["hubs"]:
        models[f"lho_{h}"] = joblib.load(MODELS / f"{args.run_id}_lho_{h}.joblib")
    tr = DD.labelled(DD.flag_groups(DD.load_block("train"), cfg))
    tr_ok = tr[~tr.held_group]
    bar_main = BL.fit(tr_ok)
    bar_lho = {h: BL.fit(tr_ok[tr_ok.hub != h]) for h in cfg["hubs"]}
    if args.block in ("test", "validation"):
        from rail_risk.baselines.split import network_split

        split, _ = network_split()
        days = [str(d) for d in (split.test if args.block == "test" else split.validation[:3])]
        df = score_days(models, pick, days, args.block, cfg)
    else:
        days = [str(d) for d in B.february_days()]
        df = score_days(models, pick, days, "february", cfg)
    assert df["ssd"].nunique() == len(days)
    if eta > 0:
        t, r = O.seconds(df["snap_time"]), O.seconds(df["act_dep_out"])
        for k in list(df.columns):
            if k == "p_main":
                df[k], _ = O.run(
                    df["hub"].to_numpy(), t, df[k].to_numpy(), df["miss"].to_numpy(float), r, eta
                )
    df["p_bar"] = BL.predict(bar_main, df)
    held = df[df["held_group"]]
    res = {}
    res.update(
        table(
            held,
            held["p_main"].to_numpy(),
            held["p_bar"].to_numpy(),
            cfg,
            "all_pairs",
            "held",
            res_n,
            seed,
        )
    )
    res.update(
        table(
            held,
            held["p_main"].to_numpy(),
            held["p_bar"].to_numpy(),
            cfg,
            "not_arrived",
            "held",
            res_n,
            seed,
        )
    )
    for h in cfg["hubs"]:
        sub = df[df["hub"] == h]
        pb = BL.predict(bar_lho[h], sub)
        for subset in ("all_pairs", "not_arrived"):
            part = table(
                sub,
                sub[f"p_lho_{h}"].to_numpy(),
                pb,
                {"hubs": [h]},
                subset,
                f"lho_{h}",
                res_n,
                seed,
            )
            res.update({k: v for k, v in part.items() if f"|{h}|" in k})
    orig = original_pairs(df, args.block)
    orig_res = {}
    for (hub, outb), g in orig.groupby(["hub", "outbound"]):
        for subset in ("all_pairs", "not_arrived"):
            gg = g[g["nya"]] if subset == "not_arrived" else g
            for hz in (*HORIZONS, "all"):
                m = gg if hz == "all" else gg[gg["horizon_min"] == hz]
                if len(m) < 30:
                    orig_res[f"{hub}|{outb}|{subset}|{hz}"] = {
                        "n": int(len(m)),
                        "skipped": "fewer than 30",
                    }
                    continue
                orig_res[f"{hub}|{outb}|{subset}|{hz}"] = cell(
                    m, m["p_main"].to_numpy(), m["p_bar"].to_numpy(), res_n, seed, ece=False
                )
    counts = {}
    for sub_name, sub in (("all_groups", df), ("held_groups", held)):
        for hub in cfg["hubs"]:
            for hz in HORIZONS:
                m = (sub["hub"] == hub) & (sub["horizon_min"] == hz)
                counts[f"{sub_name}|{hub}|{hz}"] = {
                    "pairs": int(m.sum()),
                    "not_arrived": int((m & sub["nya"]).sum()),
                }
    out = {"block": args.block, "run_id": args.run_id, "git_head": head, "git_dirty": dirty,
           "option": pick, "online_eta": eta, "dates": len(days), "rows": len(df),
           "verdict_held_out_groups": verdict(res, cfg), "cells": res, "original_pairs": orig_res,
           "counts": counts}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    if dry:
        print(json.dumps(out["verdict_held_out_groups"], indent=1))
        (OUT / "validation_dryrun_results.json").write_text(
            json.dumps(out, indent=1, default=float)
        )
        return
    (OUT / f"{args.block}_results.json").write_text(json.dumps(out, indent=1, default=float))
    df.to_parquet(MODELS / f"{args.run_id}_{args.block}_predictions.parquet", index=False)
    mlflow.set_tracking_uri(f"sqlite:///{Path('mlruns/mlflow.db').resolve()}")
    with mlflow.start_run(run_id=args.run_id):
        for k, v in out["verdict_held_out_groups"].items():
            if isinstance(v, dict):
                mlflow.log_metric(f"{args.block}_{k}_pass", float(v["pass"]))
        mlflow.log_metric(
            f"{args.block}_hubs_passing", float(out["verdict_held_out_groups"]["hubs_passing"])
        )
        for k, v in res.items():
            if k.startswith("held|not_arrived|") and k.endswith("|all") and "paired_brier" in v:
                hub = k.split("|")[2]
                mlflow.log_metrics({f"{args.block}_{hub}_brier_diff_nya": v["paired_brier"]["diff"],
                                    f"{args.block}_{hub}_ece_nya": v["ece"]["value"]})  # fmt: skip
        mlflow.set_tag(f"{args.block}_scored_at_head", f"{head}{'+dirty' if dirty else ''}")
    marker.write_text(
        json.dumps({"run_id": args.run_id, "git_head": head, "dirty": dirty, "rows": len(df)})
        + "\n"
    )
    print(json.dumps(out["verdict_held_out_groups"], indent=1))


if __name__ == "__main__":
    main()
