"""LightGBM quantile models with conformalized quantile regression (CQR).

One model per quantile, with the horizon as a feature. The conformal offset is computed on a
calibration block separately per horizon, and for the Mondrian variant also per observed
lateness group at T, then applied to the 0.1 to 0.9 band so that it reaches the nominal
coverage within each group. Quantiles are sorted per row so they never cross.

The target is either the destination delay (direct) or that delay minus the lateness at T
(residual), so the persistence forecast is the floor the residual model starts from.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from rail_risk.features.snapshots import FEATURE_COLUMNS
from rail_risk.lateness import DEFAULT_EDGES, lateness_base, lateness_groups, merge_groups
from rail_risk.logging import get_logger

TARGETS = ("direct", "residual")
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
    target: str = "direct"
    global_offsets: dict[int, float] = field(default_factory=dict)
    offsets: dict[tuple[int, int], float] = field(default_factory=dict)  # (horizon, group)
    group_map: dict[int, dict[int, int]] = field(default_factory=dict)  # horizon -> group -> rep
    merges: list[dict[str, Any]] = field(default_factory=list)
    edges: tuple[float, ...] = DEFAULT_EDGES

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        """Sorted quantiles on the delay scale. A residual model adds the lateness at T back."""
        x = self.design.transform(df)
        preds = np.column_stack([self.boosters[q].predict(x) for q in self.quantiles])
        if self.target == "residual":
            preds = preds + lateness_base(df)[:, None]
        return sort_quantiles(preds)

    def predict(
        self, df: pd.DataFrame, mode: str = "mondrian"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(point, lower, upper) after the conformal adjustment for each row's horizon.

        mode "mondrian" uses the offset of the row's lateness group, "global" the horizon's.
        """
        raw = self.raw(df)
        lo, med, hi = raw[:, 0], raw[:, len(self.quantiles) // 2], raw[:, -1]
        horizons = df["horizon_min"].to_numpy()
        if mode == "global":
            offset = np.array([self.global_offsets[int(h)] for h in horizons])
        else:
            groups = lateness_groups(df, self.edges)
            offset = np.array(
                [
                    self.offsets[(int(h), self.group_map[int(h)][int(g)])]
                    for h, g in zip(horizons, groups, strict=True)
                ]
            )
        new_lo, new_hi = apply_offset(lo, med, hi, offset)
        return med, new_lo, new_hi

    def calibrate(
        self, df: pd.DataFrame, edges: tuple[float, ...] = DEFAULT_EDGES, min_rows: int = 30
    ) -> "QuantileFit":
        """Offsets per horizon (global) and per horizon and lateness group (Mondrian).

        A group with fewer than `min_rows` calibration rows is merged into its neighbour.
        """
        log = get_logger("conformal")
        self.edges = tuple(edges)
        raw = self.raw(df)
        y_all = df["label_delay_min"].to_numpy(dtype=float)
        groups_all = lateness_groups(df, self.edges)
        for h, idx in df.groupby("horizon_min").indices.items():
            h = int(h)
            y, lo, hi, g = y_all[idx], raw[idx, 0], raw[idx, -1], groups_all[idx]
            self.global_offsets[h] = conformal_offset(y, lo, hi, self.coverage)
            counts = {k: int((g == k).sum()) for k in range(len(self.edges) + 2)}
            mapping, merges = merge_groups(counts, min_rows)
            self.group_map[h] = mapping
            for small, into in merges:
                record = {"horizon": h, "merged": small, "into": into, "counts": counts}
                self.merges.append(record)
                log.info("lateness_group_merged", target=self.target, **record)
            for rep in sorted(set(mapping.values())):
                mask = np.isin(g, [k for k, v in mapping.items() if v == rep])
                self.offsets[(h, rep)] = conformal_offset(
                    y[mask], lo[mask], hi[mask], self.coverage
                )
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


def _target_values(df: pd.DataFrame, target: str) -> np.ndarray:
    """What the models learn: the delay, or the delay minus the lateness at T."""
    y = df["label_delay_min"].to_numpy(dtype=float)
    return y - lateness_base(df) if target == "residual" else y


def grid_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in product(*grid.values())]


def fit_quantile(
    train: pd.DataFrame,
    tune: pd.DataFrame,
    calibrate: pd.DataFrame,
    cfg: dict[str, Any],
    columns: list[str] | None = None,
    target: str = "direct",
) -> tuple[QuantileFit, list[dict[str, Any]]]:
    """Grid search by validation pinball, early stopping on the median model, then CQR.

    The lower and upper models reuse the median model's best iteration count. Returns the fit
    and the grid results. `tune` and `calibrate` are the same frame unless validation was split.
    With target "residual" the models learn the delay minus the lateness at T, and `raw` adds
    it back, so every score and offset is on the delay scale.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    quantiles = tuple(cfg["quantiles"])
    design = Design(columns).fit(train)
    x_train = design.transform(train)
    y_train = _target_values(train, target)
    x_tune = design.transform(tune)
    y_tune = _target_values(tune, target)
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
                QuantileFit(
                    design, quantiles, boosters, params, rounds, cfg["nominal_coverage"], target
                ),
            )
    assert best is not None
    return best[1].calibrate(calibrate, *_mondrian(cfg)), results


def _mondrian(cfg: dict[str, Any]) -> tuple[tuple[float, ...], int]:
    m = cfg.get("mondrian", {})
    return tuple(m.get("edges", DEFAULT_EDGES)), int(m.get("min_rows", 30))


@dataclass
class QuantileModel:
    """One fit per target. Each horizon uses the target with the lower tuning pinball loss."""

    fits: dict[str, QuantileFit]
    choice: dict[int, str]
    tune_pinball: dict[str, dict[int, float]]

    def predict(
        self, df: pd.DataFrame, mode: str = "mondrian"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point, lo, hi = (np.empty(len(df)) for _ in range(3))
        for h, idx in df.groupby("horizon_min").indices.items():
            p, low, up = self.fits[self.choice[int(h)]].predict(df.iloc[idx], mode)
            point[idx], lo[idx], hi[idx] = p, low, up
        return point, lo, hi

    @property
    def chosen(self) -> QuantileFit:
        """The fit used for the most horizons, for the parameters logged to MLflow."""
        names = list(self.choice.values())
        return self.fits[max(self.fits, key=names.count)]


def fit_models(
    train: pd.DataFrame,
    tune: pd.DataFrame,
    calibrate: pd.DataFrame,
    cfg: dict[str, Any],
    columns: list[str] | None = None,
) -> tuple[QuantileModel, dict[str, list[dict[str, Any]]]]:
    """Fit each candidate target, then keep the better one per horizon by tuning pinball loss."""
    fits, grids, losses = {}, {}, {}
    for target in cfg.get("targets", list(TARGETS)):
        fit, grid = fit_quantile(train, tune, calibrate, cfg, columns, target)
        fits[target], grids[target] = fit, grid
        raw = fit.raw(tune)
        y = tune["label_delay_min"].to_numpy(dtype=float)
        losses[target] = {}
        for h, idx in tune.groupby("horizon_min").indices.items():
            losses[target][int(h)] = float(
                np.mean([pinball(y[idx], raw[idx, k], q) for k, q in enumerate(fit.quantiles)])
            )
    horizons = sorted({h for per in losses.values() for h in per})
    choice = {
        h: min(fits, key=lambda t, h=h: (losses[t].get(h, np.inf), list(fits).index(t)))
        for h in horizons
    }
    return QuantileModel(fits, choice, losses), grids
