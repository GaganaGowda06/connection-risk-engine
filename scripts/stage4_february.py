#!/usr/bin/env python3
"""February 2026 stress report (GOAL: scored once, reported separately, never used to tune).

Builds snapshots and features for the 28 complete February days through the guarded reader
(reading_split), then scores the frozen calibrated cold-start model against the Stage 2 baselines.
Nothing is fitted, tuned or calibrated on February. Refuses to run twice (marker).

    uv run python scripts/stage4_february.py --run-id RUN        (RUN is the model run id)
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.baselines import build, models, scoring
from rail_risk.baselines.split import is_complete, manifest_files
from rail_risk.baselines.target_check import assert_targets_ok
from rail_risk.clean.run import route_specs
from rail_risk.config import load_routes
from rail_risk.ingest.darwin_pport import load_stage1_config, split_for
from rail_risk.ingest.pport_extract import reading_split
from rail_risk.network_model import calibrate as C
from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F
from rail_risk.network_model import run as R
from rail_risk.network_model.evaluate import paired_brier, paired_mae_diff

OUT = Path("reports/network_model/stage4")
MARKER = OUT / "february_scored.json"
CAL_DIR = Path("data/interim/stage4/calibrated")
FEB = Path("data/interim/stage4/february")
HEADLINE = (15, 60)


def february_days() -> list[date]:
    cfg = load_stage1_config()
    files = manifest_files(Path("data/manifests/darwin_archive.csv"))
    days = [date(2026, 2, d) for d in range(1, 29)]
    return [d for d in days if split_for(d, cfg) == "winter_test" and is_complete(d, files)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--mlflow-run-id", required=True)
    args = ap.parse_args()
    if MARKER.exists():
        raise SystemExit(f"February was already scored: {MARKER.read_text()[:200]}")
    days = february_days()
    cal = C.Calibrated.load(CAL_DIR / args.run_id)
    # training rows for the baselines (train block only), read before February is opened
    rows, held, split, _ = R.prepare_frames(include_test=False, cold=True)
    cold_fit = models.fit(rows["train"])
    hist_fit = models.fit(rows["train_all_routes"])
    routes = load_routes("config/routes.yaml")
    table = [r for r in routes.routes if not r.legacy]
    stations = pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
    specs = route_specs(type(routes)(routes=table), stations)
    with reading_split("winter_test", "february_stress"):
        built = build.build_range(D.PPORT, days, specs, FEB / "snapshots", rebuild=True)
        feats = F.build_range(D.PPORT, FEB / "snapshots", FEB / "features", days, rebuild=True)
    fastest = pd.read_parquet(D.FASTEST_COLD)
    df = D.prepare(D.load(days, FEB / "features"), fastest)
    df = D.add_duration(df, FEB / "snapshots")
    df = df[df["has_target"]].reset_index(drop=True)
    target_report = assert_targets_ok(None, {"february": df})
    pred = cal.predict(df, D.base_point(cold_fit, df))
    bars = {"cold": models.predict(cold_fit, df), "history": models.predict(hist_fit, df)}
    y = df["target_min"].to_numpy()
    is_held = df["route"].isin(held).to_numpy()
    out: dict = {
        "run_id": args.run_id, "days": len(days), "rows": len(df), "build": built,
        "features": feats, "target_check": target_report, "held_out_rows": int(is_held.sum()),
    }  # fmt: skip
    groups = {"held_out": is_held, "train_routes": ~is_held, "all": np.ones(len(df), bool)}
    methods = {"model": pred[["point", "lo", "hi", "p_late"]]}
    for bar, preds in bars.items():
        for name in ("persistence", "shrunk_persistence", "historical"):
            methods[f"{name}_{bar}"] = preds[name]
    stats = {k: scoring.row_stats(df.assign(schedule_known=True), v) for k, v in methods.items()}
    sc = scoring.score_all(stats, held, sorted(df["route"].unique()), 1000, 20261015, 0.95)
    sc.to_csv(OUT / "february_scores.csv", index=False)
    summary = {}
    for g in groups:
        for k in methods:
            for h in HEADLINE:
                r = sc[(sc.method == k) & (sc.group == g) & (sc.horizon_min == h)]
                if r.empty:
                    continue
                r = r.iloc[0]
                summary[f"{g}/{k}/{h}"] = {
                    c: round(float(r[c]), 4)
                    for c in ("n", "mae", "mae_lo", "mae_hi", "coverage", "brier", "ece")
                }
    out["scores"] = summary
    out["coverage_calibrated_held_out"] = C.coverage_table(
        df[is_held], pred["lo"].to_numpy()[is_held], pred["hi"].to_numpy()[is_held], HEADLINE
    )
    out["coverage_raw_held_out"] = C.coverage_table(
        df[is_held],
        pred["raw_lo"].to_numpy()[is_held],
        pred["raw_hi"].to_numpy()[is_held],
        HEADLINE,
    )
    paired = []
    for gname, mask in (("held_out", is_held), ("all_routes", np.ones(len(df), bool))):
        for bar, preds in bars.items():
            for name in ("persistence", "shrunk_persistence", "historical"):
                if bar == "history" and name == "persistence":
                    continue
                for h in HEADLINE:
                    m = mask & (df["horizon_min"].to_numpy() == h)
                    r = paired_mae_diff(
                        pred["point"].to_numpy()[m] - y[m],
                        preds[name]["point"].to_numpy()[m] - y[m],
                        df["ssd"].to_numpy()[m], 1000, 20261015,
                    )  # fmt: skip
                    paired.append({"group": gname, "bar": bar, "baseline": name,
                                   "horizon_min": h, "n": int(m.sum()), **r})  # fmt: skip
    out["paired_mae"] = paired
    held_df = df[is_held].reset_index(drop=True)
    pl = pred[is_held].reset_index(drop=True)
    out["brier_paired_held_out"] = {
        f"vs_historical_{bar}_{h}": paired_brier(
            held_df, pl["p_late"].to_numpy(),
            bars[bar]["historical"]["p_late"].to_numpy()[is_held], h, 1000, 20261015,
        )
        for bar in bars
        for h in HEADLINE
    }  # fmt: skip
    rt = []
    for route in sorted(held):
        for h in HEADLINE:
            m = ((df["route"] == route) & (df["horizon_min"] == h)).to_numpy()
            if not m.any():
                continue
            ok = (pred["lo"].to_numpy()[m] <= y[m]) & (y[m] <= pred["hi"].to_numpy()[m])
            rt.append({"route": route, "horizon_min": h, "n": int(m.sum()),
                       "mae": float(np.abs(pred["point"].to_numpy()[m] - y[m]).mean()),
                       "coverage": float(ok.mean())})  # fmt: skip
    out["per_route"] = rt
    (OUT / "february_results.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    head, dirty = R.git_state()
    MARKER.write_text(json.dumps({"run_id": args.run_id, "git_head": head, "dirty": dirty}) + "\n")
    import mlflow

    R._tracking()
    with mlflow.start_run(run_id=args.mlflow_run_id):
        mlflow.set_tag("february_scored_at_head", f"{head}{'+dirty' if dirty else ''}")
        for k in ("held_out/model/15", "held_out/model/60"):
            for c in ("mae", "coverage", "ece"):
                mlflow.log_metric(f"feb_{k.replace('/', '_')}_{c}", summary[k][c])
    print(
        json.dumps({k: out[k] for k in ("days", "rows", "held_out_rows", "target_check")}, indent=1)
    )


if __name__ == "__main__":
    main()
