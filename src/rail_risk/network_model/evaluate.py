"""Paired date-bootstrap comparisons and the score tables for the route-agnostic model.

The unit of resampling is the service date (as in Stage 2): per date we keep the number of rows
and the sum of the per-row absolute-error difference, then resample dates with replacement. The
difference is model minus baseline, so a 95% CI entirely below zero means the model is better.
"""

import numpy as np
import pandas as pd


def paired_mae_diff(
    err_model: np.ndarray,
    err_base: np.ndarray,
    ssd: np.ndarray,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Mean of (|model error| - |baseline error|) with a date-bootstrap CI."""
    d = np.abs(err_model) - np.abs(err_base)
    dates, idx = np.unique(ssd, return_inverse=True)
    n = np.bincount(idx).astype(float)
    s = np.bincount(idx, weights=d)
    if len(dates) < 2:
        return {"diff": float(d.mean()), "lo": float("nan"), "hi": float("nan"), "n_dates": 1}
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(dates), np.full(len(dates), 1 / len(dates)), size=resamples)
    boot = (counts @ s) / (counts @ n)
    a = (1 - confidence) / 2
    return {
        "diff": float(s.sum() / n.sum()),
        "lo": float(np.quantile(boot, a)),
        "hi": float(np.quantile(boot, 1 - a)),
        "n_dates": int(len(dates)),
    }


def paired_table(
    frame: pd.DataFrame,
    model_pred: np.ndarray,
    baselines: dict[str, np.ndarray],
    horizons: tuple[int, ...],
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """One row per baseline and horizon: model MAE, baseline MAE, paired difference and CI."""
    y = frame["target_min"].to_numpy()
    h = frame["horizon_min"].to_numpy()
    rows = []
    for name, base in baselines.items():
        for horizon in horizons:
            m = h == horizon
            if not m.any():
                continue
            e_model, e_base = model_pred[m] - y[m], base[m] - y[m]
            res = paired_mae_diff(
                e_model, e_base, frame["ssd"].to_numpy()[m], resamples, seed, confidence
            )
            rows.append(
                {
                    "baseline": name,
                    "horizon_min": horizon,
                    "n": int(m.sum()),
                    "model_mae": float(np.abs(e_model).mean()),
                    "baseline_mae": float(np.abs(e_base).mean()),
                    **res,
                    "ci_below_zero": bool(res["hi"] < 0),
                }
            )
    return pd.DataFrame(rows)


def paired_brier(
    frame: pd.DataFrame,
    p_model: np.ndarray,
    p_base: np.ndarray,
    horizon: int,
    resamples: int,
    seed: int,
    late_min: float = 15.0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Mean of (model Brier - baseline Brier) for P(delay >= late_min) with a date-bootstrap CI."""
    m = frame["horizon_min"].to_numpy() == horizon
    late = (frame["target_min"].to_numpy()[m] >= late_min).astype(float)
    d = (p_model[m] - late) ** 2 - (p_base[m] - late) ** 2
    dates, idx = np.unique(frame["ssd"].to_numpy()[m], return_inverse=True)
    n, s = np.bincount(idx).astype(float), np.bincount(idx, weights=d)
    counts = np.random.default_rng(seed).multinomial(
        len(dates), np.full(len(dates), 1 / len(dates)), size=resamples
    )
    boot = (counts @ s) / (counts @ n)
    a = (1 - confidence) / 2
    return {
        "diff": float(s.sum() / n.sum()),
        "lo": float(np.quantile(boot, a)),
        "hi": float(np.quantile(boot, 1 - a)),
    }
