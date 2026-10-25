"""Stage 4 helpers: the out-of-route calibration pool and the method comparison.

The pool holds predictions on rows a model did not train on by route: the tuning-fold models
scored on the routes each fold left out (train block), and the final model on validation rows of
the training routes. No held-out route is in it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.baselines import models
from rail_risk.network_model import calibrate as C
from rail_risk.network_model import dataset as D
from rail_risk.network_model import train as T

POOL_COLS = [
    "ssd", "route", "horizon_min", "target_min", "late_now_min", "stops_remaining",
    "sched_duration_min", "n_actuals", "ahead_n", "dest_n", "next_n",
]  # fmt: skip
POOL_DIR = Path("data/interim/stage4")


def fold_pool(
    rows_train: pd.DataFrame, cfg: dict, tuned: dict, base_rows: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Leave-route-group-out predictions over every train day, one fold model per route group.
    base_rows (history mode) is the table the base is fitted on, fold routes' history included;
    without it the fold's base is fitted on the fold's training routes only (cold start)."""
    folds = T.route_folds(sorted(rows_train["route"].unique()), cfg["folds"])
    parts = []
    for fold in folds:
        va = rows_train[rows_train["route"].isin(fold)]
        tr = rows_train[~rows_train["route"].isin(fold)]
        fitted = models.fit(tr if base_rows is None else base_rows)
        warm = cfg.get("warm_base", False)
        base_tr = D.base_point(fitted, tr, warm)
        base_va = D.base_point(fitted, va, warm)
        boosters = T.fit_boosters(tr, cfg, tuned, base_tr)
        pred = boosters.predict(va, base_va)
        parts.append(pd.concat([va[POOL_COLS], pred[["point", "lo", "hi", "p_late_raw"]]], axis=1))
    return pd.concat(parts, ignore_index=True)


def final_pool_rows(
    model: T.Booster3,
    base_rows: pd.DataFrame,
    val_train_routes: pd.DataFrame,
    warm: bool = False,
) -> pd.DataFrame:
    """The final model on validation rows of the training routes (same base as training)."""
    fitted = models.fit(base_rows)
    base = D.base_point(fitted, val_train_routes, warm)
    pred = model.predict(val_train_routes, base)
    return pd.concat(
        [val_train_routes[POOL_COLS], pred[["point", "lo", "hi", "p_late_raw"]]], axis=1
    ).reset_index(drop=True)


def compare_methods(pool: pd.DataFrame, held_val: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Coverage, worst route and width on held-out validation rows, for each range method."""
    out = {"raw": C.coverage_table(held_val, pred["lo"].to_numpy(), pred["hi"].to_numpy())}
    for method in C.METHODS:
        cal = C.RangeCalibrator(method).fit(pool)
        lo, hi = cal.apply(
            held_val, pred["point"].to_numpy(), pred["lo"].to_numpy(), pred["hi"].to_numpy()
        )
        out[method] = C.coverage_table(held_val, lo, hi)
        out[method]["fallback_rows"] = cal.fallback_rows
    return out


def brief(table: dict) -> dict:
    """The numbers used to choose: pooled coverage, per-horizon coverage, worst route, width."""
    return {
        "overall": round(table["overall"], 3),
        "cov15": round(table[15]["coverage"], 3),
        "cov60": round(table[60]["coverage"], 3),
        "worst15": f"{table[15]['worst_route']} {table[15]['worst_route_coverage']:.3f}",
        "worst60": f"{table[60]['worst_route']} {table[60]['worst_route_coverage']:.3f}",
        "width": round(table["mean_width"], 2),
    }


def nan_safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def prepare_calibration_frames(cold: bool = True):
    """Frames with the length column, from run.prepare_frames (validation, never the test)."""
    from rail_risk.network_model import run as R

    rows, held, split, _ = R.prepare_frames(include_test=False, cold=cold)
    for k in ("train", "train_all_routes", "validation"):
        rows[k] = D.add_duration(rows[k])
    return rows, held, split
