#!/usr/bin/env python3
"""Stage 4 step 4: score the test block once with the frozen calibrated model, held-out routes
only. Refuses to run twice (marker). Writes reports/network_model/stage4/test_*.

    uv run python scripts/stage4_score_test.py --run-id RUN        (RUN is the model run id)
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from rail_risk.baselines import models, scoring
from rail_risk.network_model import calibrate as C
from rail_risk.network_model import dataset as D
from rail_risk.network_model import run as R
from rail_risk.network_model.evaluate import paired_brier, paired_mae_diff

OUT = Path("reports/network_model/stage4")
MARKER = OUT / "test_scored.json"
CAL_DIR = Path("data/interim/stage4/calibrated")
HEADLINE = (15, 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--mlflow-run-id", required=True)
    args = ap.parse_args()
    if MARKER.exists():
        raise SystemExit(f"the test block was already scored: {MARKER.read_text()[:200]}")
    cal = C.Calibrated.load(CAL_DIR / args.run_id)
    rows, held, split, _ = R.prepare_frames(include_test=True, cold=True)
    for k in ("train", "train_all_routes", "test"):
        rows[k] = D.add_duration(rows[k])
    hold = rows["test"][rows["test"]["route"].isin(held)].reset_index(drop=True)
    cold_fit = models.fit(rows["train"])
    hist_fit = models.fit(rows["train_all_routes"])  # the Stage 2 bar, with route history
    pred = cal.predict(hold, D.base_point(cold_fit, hold))
    bars = {"cold": models.predict(cold_fit, hold), "history": models.predict(hist_fit, hold)}
    y = hold["target_min"].to_numpy()
    out: dict = {"rows": len(hold), "dates": int(hold["ssd"].nunique()), "run_id": args.run_id}

    # test 3: coverage
    out["coverage"] = {
        "raw": C.coverage_table(
            hold, pred["raw_lo"].to_numpy(), pred["raw_hi"].to_numpy(), HEADLINE
        ),
        "calibrated": C.coverage_table(
            hold, pred["lo"].to_numpy(), pred["hi"].to_numpy(), HEADLINE
        ),
    }
    allh = tuple(sorted(hold["horizon_min"].unique()))
    out["coverage_all_horizons"] = {
        h: C.coverage_table(hold, pred["lo"].to_numpy(), pred["hi"].to_numpy(), (h,))[h]["coverage"]
        for h in allh
    }
    # test 2 context: paired MAE against persistence, shrunk persistence, historical in both bars
    paired = []
    for bar, preds in bars.items():
        for name in ("persistence", "shrunk_persistence", "historical"):
            if bar == "history" and name == "persistence":
                continue
            for h in HEADLINE:
                m = hold["horizon_min"].to_numpy() == h
                r = paired_mae_diff(
                    pred["point"].to_numpy()[m] - y[m], preds[name]["point"].to_numpy()[m] - y[m],
                    hold["ssd"].to_numpy()[m], 1000, 20261015,
                )  # fmt: skip
                paired.append(
                    {"bar": bar, "baseline": name, "horizon_min": h, "n": int(m.sum()),
                     "model_mae": float(np.abs(pred["point"].to_numpy()[m] - y[m]).mean()),
                     "baseline_mae": float(
                         np.abs(preds[name]["point"].to_numpy()[m] - y[m]).mean()
                     ),
                     **r}
                )  # fmt: skip
    out["paired_mae"] = paired
    # test 4: Brier and ECE of P(delay >= 15)
    frame = hold.assign(schedule_known=True)
    sc = {}
    methods = {"model": pred[["point", "lo", "hi", "p_late"]]}
    for bar, preds in bars.items():
        for name in ("historical", "persistence"):
            methods[f"{name}_{bar}"] = preds[name]
    stats = {k: scoring.row_stats(frame, v) for k, v in methods.items()}
    table = scoring.score_all(stats, held, sorted(hold["route"].unique()), 1000, 20261015, 0.95)
    table.to_csv(OUT / "test_scores.csv", index=False)
    for k in methods:
        for h in HEADLINE:
            r = table[(table.method == k) & (table.group == "held_out") & (table.horizon_min == h)]
            sc[f"{k}_{h}"] = {
                c: float(r.iloc[0][c])
                for c in ("brier", "brier_lo", "brier_hi", "ece", "ece_lo", "ece_hi")
            }
    out["p_late"] = sc
    out["brier_paired"] = {
        f"vs_{b}_{h}": paired_brier(
            hold, pred["p_late"].to_numpy(), methods[b]["p_late"].to_numpy(), h, 1000, 20261015
        )
        for b in ("historical_history", "historical_cold", "persistence_cold")
        for h in HEADLINE
    }
    # per-route table with widths
    rt = []
    for route in sorted(held):
        for h in HEADLINE:
            m = ((hold["route"] == route) & (hold["horizon_min"] == h)).to_numpy()
            cov = lambda lo, hi, m=m: float(((lo[m] <= y[m]) & (y[m] <= hi[m])).mean())  # noqa: E731
            rt.append(
                {"route": route, "horizon_min": h, "n": int(m.sum()),
                 "mae": float(np.abs(pred["point"].to_numpy()[m] - y[m]).mean()),
                 "raw_coverage": cov(pred["raw_lo"].to_numpy(), pred["raw_hi"].to_numpy()),
                 "coverage": cov(pred["lo"].to_numpy(), pred["hi"].to_numpy()),
                 "raw_width": float((pred["raw_hi"] - pred["raw_lo"]).to_numpy()[m].mean()),
                 "width": float((pred["hi"] - pred["lo"]).to_numpy()[m].mean())}
            )  # fmt: skip
    out["per_route"] = rt
    pd.DataFrame(rt).to_csv(OUT / "test_per_route.csv", index=False)
    keep = ["ssd", "rid", "route", "horizon_min", "target_min", "point", "lo", "hi", "p_late"]
    keep += ["late_now_min", "n_actuals", "ahead_n", "dest_n", "next_n", "stops_remaining"]
    keep += ["sched_duration_min"]
    hold.assign(point=pred["point"], lo=pred["lo"], hi=pred["hi"], p_late=pred["p_late"])[
        keep
    ].to_parquet(CAL_DIR / args.run_id / "test_predictions.parquet", index=False)
    (OUT / "test_results.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    head, dirty = R.git_state()
    MARKER.write_text(json.dumps({"run_id": args.run_id, "git_head": head, "dirty": dirty}) + "\n")
    R._tracking()
    with mlflow.start_run(run_id=args.mlflow_run_id):
        mlflow.set_tag("test_scored_at_head", f"{head}{'+dirty' if dirty else ''}")
        for r in rt:
            mlflow.log_metric(f"test_cov_{r['route']}_{r['horizon_min']}", r["coverage"])
        for k, v in out["coverage"]["calibrated"].items():
            if isinstance(v, float):
                mlflow.log_metric(f"test_{k}", v)
    print(
        json.dumps(
            {k: out[k] for k in ("coverage", "p_late", "brier_paired")}, indent=1, default=str
        )[:6000]
    )


if __name__ == "__main__":
    main()
