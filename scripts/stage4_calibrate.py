#!/usr/bin/env python3
"""Stage 4 step 3: fit the chosen range method and the P(late) calibrators on out-of-route data
(tuning-fold predictions plus validation rows of the training routes), report ECE before and
after on validation held-out routes, save the calibrated model and track it in MLflow.

    uv run python scripts/stage4_calibrate.py --run-id RUN [--method cqr]
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from rail_risk.baselines import models
from rail_risk.network_model import calibrate as C
from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F
from rail_risk.network_model import run as R
from rail_risk.network_model import stage4 as S
from rail_risk.network_model import train as T

OUT = Path("reports/network_model/stage4")
CAL_DIR = Path("data/interim/stage4/calibrated")
NAME = "cold_noslack"
DROP = [*F.SLACK, *F.HISTORY]


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    b = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / len(p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--method", default="cqr", choices=list(C.METHODS))
    args = ap.parse_args()
    cfg = {**T.load_config(), "drop": DROP, "warm_base": False}
    tuned = json.loads((OUT / NAME / "tuning.json").read_text())
    rows, held, split = S.prepare_calibration_frames(cold=True)
    train, val = rows["train"], rows["validation"]
    model = T.Booster3.load(T.MODEL_DIR / args.run_id)
    cache = S.POOL_DIR / f"fold_pool_{args.run_id}.parquet"
    if cache.exists():
        fold = pd.read_parquet(cache)
    else:
        fold = S.fold_pool(train, cfg, tuned)
        fold.to_parquet(cache, index=False)
    own = S.final_pool_rows(model, train, val[~val["route"].isin(held)])
    pool = pd.concat([fold, own], ignore_index=True)
    ranges = C.RangeCalibrator(args.method).fit(pool)
    # P(delay >= 15): isotonic per horizon on out-of-route raw scores
    pl = {}
    from sklearn.isotonic import IsotonicRegression

    late = (pool["target_min"] >= models.LATE_MIN).astype(float).to_numpy()
    for k in sorted(pool["horizon_min"].unique()):
        m = (pool["horizon_min"] == k).to_numpy()
        pl[int(k)] = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            pool["p_late_raw"].to_numpy()[m], late[m]
        )
    cal = C.Calibrated(
        model, ranges, pl,
        {"model_run_id": args.run_id, "method": args.method, "pool_rows": len(pool),
         "fold_rows": len(fold), "validation_rows": len(own), "mode": NAME},
    )  # fmt: skip
    hv = val[val["route"].isin(held)].reset_index(drop=True)
    base = D.base_point(models.fit(train), hv)
    pred = cal.predict(hv, base)
    y = (hv["target_min"] >= models.LATE_MIN).astype(float).to_numpy()
    res = {
        "method": args.method,
        "pool": {k: cal.meta[k] for k in ("pool_rows", "fold_rows", "validation_rows")},
    }
    res["ranges_validation_held_out"] = {
        "raw": S.brief(C.coverage_table(hv, pred["raw_lo"].to_numpy(), pred["raw_hi"].to_numpy())),
        "calibrated": S.brief(C.coverage_table(hv, pred["lo"].to_numpy(), pred["hi"].to_numpy())),
    }
    res["p_late_validation_held_out"] = {}
    for h in (15, 60):
        m = (hv["horizon_min"] == h).to_numpy()
        before, raw = pred["p_late_stage3"].to_numpy()[m], pred["p_late_raw"].to_numpy()[m]
        after = pred["p_late"].to_numpy()[m]
        res["p_late_validation_held_out"][h] = {
            "ece_before_validation_trainroutes_isotonic": round(ece(before, y[m]), 4),
            "ece_raw_classifier": round(ece(raw, y[m]), 4),
            "ece_after_out_of_route_isotonic": round(ece(after, y[m]), 4),
            "brier_before": round(float(((before - y[m]) ** 2).mean()), 5),
            "brier_after": round(float(((after - y[m]) ** 2).mean()), 5),
        }
    path = CAL_DIR / args.run_id
    cal.save(path)
    R._tracking()
    with mlflow.start_run(run_name=f"stage4-calibration-{args.method}") as run:
        mlflow.log_params({"method": args.method, "model_run_id": args.run_id, "mode": NAME,
                           "pool_rows": len(pool), "min_cell": C.MIN_CELL})  # fmt: skip
        head, dirty = R.git_state()
        mlflow.set_tags({"git_head": head, "git_dirty": str(dirty), "dropped": ",".join(DROP),
                         "held_out_in_pool": "0"})  # fmt: skip
        for k, v in res["ranges_validation_held_out"]["calibrated"].items():
            if isinstance(v, float):
                mlflow.log_metric(f"val_{k}", v)
        for h, d in res["p_late_validation_held_out"].items():
            for k, v in d.items():
                mlflow.log_metric(f"val_{k}_{h}", v)
        mlflow.log_artifacts(str(path), artifact_path="calibrated")
        res["mlflow_run_id"] = run.info.run_id
    (OUT / "calibration_validation.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
