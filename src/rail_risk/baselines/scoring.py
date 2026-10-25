"""Score the baselines: MAE, 80% coverage, Brier and ECE for P(late >= 15), with 95% bootstrap
confidence intervals that resample service dates (a day is the unit, so one bad day is not
hidden by thousands of correlated rows)."""

import numpy as np
import pandas as pd

from rail_risk.baselines.models import LATE_MIN

BINS = 10
STAT_COLS = ["n", "abs", "cov", "wid", "brier"]


def row_stats(df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Per-row terms for one method; rows without a target are dropped."""
    y = df["target_late_s"].astype("float64").to_numpy() / 60.0
    ok = ~np.isnan(y) & df["schedule_known"].to_numpy()
    late = (y >= LATE_MIN).astype(float)
    p = pred["p_late"].to_numpy()
    out = pd.DataFrame(
        {
            "ssd": df["ssd"].to_numpy(),
            "route": df["route"].to_numpy(),
            "horizon_min": df["horizon_min"].to_numpy(),
            "abs": np.abs(pred["point"].to_numpy() - y),
            "cov": ((pred["lo"].to_numpy() <= y) & (y <= pred["hi"].to_numpy())).astype(float),
            "wid": pred["hi"].to_numpy() - pred["lo"].to_numpy(),
            "brier": (p - late) ** 2,
            "bin": np.minimum((p * BINS).astype(int), BINS - 1),
            "late": late,
            "p": p,
        }
    )
    return out[ok]


def _date_matrix(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One row per date: n, sums, then per-bin n, sum p and sum late."""
    dates = np.sort(g["ssd"].unique())
    idx = pd.Series(range(len(dates)), index=dates)
    d = idx.loc[g["ssd"]].to_numpy()
    cols = np.zeros((len(dates), 5 + 3 * BINS))
    np.add.at(cols[:, 0], d, 1.0)
    for j, c in enumerate(("abs", "cov", "wid", "brier"), start=1):
        np.add.at(cols[:, j], d, g[c].to_numpy())
    b = g["bin"].to_numpy()
    np.add.at(cols, (d, 5 + b), 1.0)
    np.add.at(cols, (d, 5 + BINS + b), g["p"].to_numpy())
    np.add.at(cols, (d, 5 + 2 * BINS + b), g["late"].to_numpy())
    return dates, cols


def _metrics(tot: np.ndarray) -> dict[str, np.ndarray]:
    """tot has shape (k, 5 + 3 * BINS) of summed statistics."""
    n = tot[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        bn = tot[:, 5 : 5 + BINS]
        bp = tot[:, 5 + BINS : 5 + 2 * BINS] / bn
        by = tot[:, 5 + 2 * BINS : 5 + 3 * BINS] / bn
        ece = np.nansum(np.abs(bp - by) * bn, axis=1) / n
        return {
            "mae": tot[:, 1] / n,
            "coverage": tot[:, 2] / n,
            "width": tot[:, 3] / n,
            "brier": tot[:, 4] / n,
            "ece": ece,
        }


def score_group(
    g: pd.DataFrame, resamples: int, rng: np.random.Generator, confidence: float
) -> dict[str, float]:
    if g.empty:
        return {"n": 0, "n_dates": 0}
    dates, cols = _date_matrix(g)
    point = {k: float(v[0]) for k, v in _metrics(cols.sum(axis=0, keepdims=True)).items()}
    counts = rng.multinomial(len(dates), np.full(len(dates), 1 / len(dates)), size=resamples)
    boot = _metrics(counts @ cols)
    alpha = (1 - confidence) / 2
    out: dict[str, float] = {"n": int(cols[:, 0].sum()), "n_dates": len(dates)}
    for k, v in point.items():
        out[k] = v
        out[f"{k}_lo"] = float(np.nanquantile(boot[k], alpha))
        out[f"{k}_hi"] = float(np.nanquantile(boot[k], 1 - alpha))
    return out


def score_all(
    per_method: dict[str, pd.DataFrame],
    held_out: set[str],
    routes: list[str],
    resamples: int,
    seed: int,
    confidence: float,
) -> pd.DataFrame:
    """Rows: method, group, horizon. Groups are each route plus held_out, train_routes and all."""
    rng = np.random.default_rng(seed)
    rows = []
    for method, st in per_method.items():
        groups = {r: st["route"] == r for r in routes}
        groups["held_out"] = st["route"].isin(held_out)
        groups["train_routes"] = ~st["route"].isin(held_out)
        groups["all"] = pd.Series(True, index=st.index)
        for horizon, sh in st.groupby("horizon_min"):
            for name, mask in groups.items():
                res = score_group(sh[mask.loc[sh.index]], resamples, rng, confidence)
                rows.append({"method": method, "group": name, "horizon_min": int(horizon), **res})
    return pd.DataFrame(rows)
