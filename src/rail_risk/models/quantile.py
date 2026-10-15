"""LightGBM quantile models with conformalized quantile regression (CQR).

One model per quantile, with the horizon as a feature. The conformal offset is computed on a
calibration block separately per horizon and applied to the 0.1 to 0.9 band so that it reaches
the nominal coverage. Quantiles are sorted per row so they never cross.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from rail_risk.features.snapshots import FEATURE_COLUMNS

CATEGORICAL = ("toc_code", "route_id")
DEFAULT_FEATURES = [*FEATURE_COLUMNS, "route_id", "horizon_min"]


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


class Design:
    """Turns snapshot rows into a model matrix. Categories are fixed at fit time."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns or DEFAULT_FEATURES
        self.categories: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> "Design":
        for col in CATEGORICAL:
            if col in self.columns:
                self.categories[col] = sorted(df[col].dropna().astype(str).unique())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            if col in self.categories:
                out[col] = pd.Categorical(df[col].astype(str), categories=self.categories[col])
            else:
                out[col] = df[col].astype(float)
        return out


def sort_quantiles(raw: np.ndarray) -> np.ndarray:
    """Non-crossing quantiles: sort each row so lower <= median <= upper."""
    return np.sort(raw, axis=1)


def conformal_offset(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, coverage: float) -> float:
    """CQR score quantile. Positive widens the band, negative narrows it.

    The level is ceil((n + 1) * coverage) / n, the finite-sample correction.
    """
    n = len(y)
    if n == 0:
        raise ValueError("no calibration rows")
    scores = np.maximum(lo - y, y - hi)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_offset(
    lo: np.ndarray, med: np.ndarray, hi: np.ndarray, offset: float
) -> tuple[np.ndarray, np.ndarray]:
    """Widen (or narrow) the band by the offset, keeping lower <= median <= upper.

    Narrowing is clipped at the median, which can only widen the band again, so the clip never
    costs coverage.
    """
    return np.minimum(lo - offset, med), np.maximum(hi + offset, med)


@dataclass
class QuantileFit:
    design: Design
    quantiles: tuple[float, ...]
    boosters: dict[float, lgb.Booster]
    params: dict[str, Any]
    best_iteration: int
    coverage: float
    offsets: dict[int, float] = field(default_factory=dict)

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        x = self.design.transform(df)
        preds = np.column_stack([self.boosters[q].predict(x) for q in self.quantiles])
        return sort_quantiles(preds)

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(point, lower, upper) after the conformal adjustment for each row's horizon."""
        raw = self.raw(df)
        lo, med, hi = raw[:, 0], raw[:, len(self.quantiles) // 2], raw[:, -1]
        offset = np.array([self.offsets[int(h)] for h in df["horizon_min"]])
        new_lo, new_hi = apply_offset(lo, med, hi, offset)
        return med, new_lo, new_hi

    def calibrate(self, df: pd.DataFrame) -> "QuantileFit":
        raw = self.raw(df)
        for h, idx in df.groupby("horizon_min").indices.items():
            y = df["label_delay_min"].to_numpy(dtype=float)[idx]
            self.offsets[int(h)] = conformal_offset(y, raw[idx, 0], raw[idx, -1], self.coverage)
        return self


def _booster(
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    q: float,
    rounds: int,
    valid: tuple[pd.DataFrame, np.ndarray] | None,
    early_stopping: int,
    seed: int,
) -> lgb.Booster:
    full = {
        "objective": "quantile",
        "alpha": q,
        "metric": "quantile",
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        **params,
    }
    train_set = lgb.Dataset(x_train, y_train, categorical_feature="auto", free_raw_data=False)
    callbacks = []
    valid_sets = []
    if valid is not None:
        valid_sets = [lgb.Dataset(valid[0], valid[1], reference=train_set)]
        callbacks = [lgb.early_stopping(early_stopping, verbose=False)]
    return lgb.train(full, train_set, rounds, valid_sets=valid_sets, callbacks=callbacks)


def grid_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in product(*grid.values())]


def fit_quantile(
    train: pd.DataFrame,
    tune: pd.DataFrame,
    calibrate: pd.DataFrame,
    cfg: dict[str, Any],
    columns: list[str] | None = None,
) -> tuple[QuantileFit, list[dict[str, Any]]]:
    """Grid search by validation pinball, early stopping on the median model, then CQR.

    The lower and upper models reuse the median model's best iteration count. Returns the fit
    and the grid results. `tune` and `calibrate` are the same frame unless validation was split.
    """
    quantiles = tuple(cfg["quantiles"])
    design = Design(columns).fit(train)
    x_train, y_train = design.transform(train), train["label_delay_min"].to_numpy(dtype=float)
    x_tune, y_tune = design.transform(tune), tune["label_delay_min"].to_numpy(dtype=float)
    fixed = {**cfg["fixed"], "num_threads": cfg["num_threads"]}
    mid = quantiles[len(quantiles) // 2]

    results = []
    best: tuple[float, QuantileFit] | None = None
    for combo in grid_combinations(cfg["grid"]):
        params = {**fixed, **combo}
        median = _booster(
            params, x_train, y_train, mid, cfg["max_rounds"], (x_tune, y_tune),
            cfg["early_stopping_rounds"], cfg["seed"],
        )  # fmt: skip
        rounds = max(1, median.best_iteration)
        boosters = {mid: median}
        for q in quantiles:
            if q != mid:
                boosters[q] = _booster(params, x_train, y_train, q, rounds, None, 0, cfg["seed"])
        loss = float(
            np.mean(
                [
                    pinball(y_tune, boosters[q].predict(x_tune, num_iteration=rounds), q)
                    for q in quantiles
                ]
            )
        )
        results.append({**combo, "best_iteration": rounds, "val_pinball": loss})
        if best is None or loss < best[0]:
            best = (
                loss,
                QuantileFit(design, quantiles, boosters, params, rounds, cfg["nominal_coverage"]),
            )
    assert best is not None
    return best[1].calibrate(calibrate), results
