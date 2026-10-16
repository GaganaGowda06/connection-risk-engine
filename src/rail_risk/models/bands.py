"""LightGBM multiclass delay-band model with a validation-chosen probability calibration."""

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from rail_risk.evaluation import Bands, band_index
from rail_risk.horizons import DEFAULT_BUCKET_EDGES, horizon_buckets
from rail_risk.models.quantile import Design

EPS = 1e-15


def normalise_rows(p: np.ndarray) -> np.ndarray:
    total = p.sum(axis=1, keepdims=True)
    uniform = np.full_like(p, 1.0 / p.shape[1])
    return np.where(total > 0, p / np.where(total > 0, total, 1.0), uniform)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p_true = np.clip(p[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.mean(np.log(p_true)))


class IsotonicOvR:
    """One isotonic regression per class on the raw probability, then rows renormalised."""

    name = "isotonic"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicOvR":
        self.models = []
        for k in range(p.shape[1]):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p[:, k], (y == k).astype(float))
            self.models.append(iso)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        cal = np.column_stack([m.predict(p[:, k]) for k, m in enumerate(self.models)])
        return normalise_rows(cal)


class TemperatureScaler:
    """Divide the log probabilities by one temperature fitted by minimising log loss."""

    name = "temperature"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        logits = np.log(np.clip(p, EPS, 1.0))

        def nll(t: float) -> float:
            return log_loss(self._softmax(logits / t), y)

        self.temperature = float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)
        return self

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self._softmax(np.log(np.clip(p, EPS, 1.0)) / self.temperature)


class Identity:
    name = "none"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Identity":
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return p


def choose_calibrator(
    p: np.ndarray, y: np.ndarray, dates: np.ndarray, max_folds: int = 5
) -> tuple[Any, dict[str, float]]:
    """Pick isotonic or temperature scaling by log loss on held-out validation dates.

    Both are fitted on the same validation rows they would be scored on, which flatters
    isotonic. So each is scored by cross-validation over whole service dates, then the winner is
    refit on all validation rows. Returns (fitted calibrator, cross-validated log loss by name).
    """
    unique = np.unique(dates)
    folds = min(max_folds, len(unique))
    scores: dict[str, list[float]] = {"isotonic": [], "temperature": []}
    if folds >= 2:
        assignment = {d: i % folds for i, d in enumerate(unique)}
        fold_of = np.array([assignment[d] for d in dates])
        for f in range(folds):
            held, rest = fold_of == f, fold_of != f
            if not held.any() or not rest.any():
                continue
            for cls in (IsotonicOvR, TemperatureScaler):
                cal = cls().fit(p[rest], y[rest])
                scores[cls.name].append(log_loss(cal.transform(p[held]), y[held]))
    cv = {k: float(np.mean(v)) if v else float("nan") for k, v in scores.items()}
    usable = {k: v for k, v in cv.items() if not np.isnan(v)}
    if not usable:  # too little data to compare: temperature scaling has one parameter
        chosen = TemperatureScaler()
    else:
        chosen = IsotonicOvR() if min(usable, key=usable.get) == "isotonic" else TemperatureScaler()
    return chosen.fit(p, y), cv


# threshold calibration: P(delay >= 15) and P(delay >= 30) on top of the calibrated band model

THRESHOLDS = (15, 30)
_CLIP = 1e-6


def exceedance(p: np.ndarray, bands: Bands, minutes: int) -> np.ndarray:
    """P(delay >= minutes) as the sum of the bands at or above that band edge."""
    return p[:, bands.lower_edges.index(minutes) :].sum(axis=1)


def rebuild_bands(e: dict[int, np.ndarray], bands: Bands) -> np.ndarray:
    """Band probabilities from exceedances, kept monotone so every band is non-negative.

    `e` maps each lower band edge above the first to P(delay >= edge). Each exceedance is capped
    by the one before it, and the differences sum to 1 by construction.
    """
    edges = list(bands.lower_edges[1:])
    capped: list[np.ndarray] = []
    prev = np.ones_like(e[edges[0]])
    for edge in edges:
        cur = np.clip(np.minimum(e[edge], prev), 0.0, 1.0)
        capped.append(cur)
        prev = cur
    cols = [1.0 - capped[0]]
    cols += [capped[i] - capped[i + 1] for i in range(len(capped) - 1)]
    cols.append(capped[-1])
    return np.column_stack(cols)


class _Platt:
    """Logistic regression on the logit of the probability. Constant if one class only."""

    name = "platt"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_Platt":
        self.constant: float | None = None
        if len(np.unique(y)) < 2:
            self.constant = (float(y.sum()) + 0.5) / (len(y) + 1.0)
            return self
        self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        return self

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        q = np.clip(p, _CLIP, 1 - _CLIP)
        return np.log(q / (1 - q)).reshape(-1, 1)

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(p), self.constant)
        return self.model.predict_proba(self._logit(p))[:, 1]


class _Isotonic:
    name = "isotonic"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_Isotonic":
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict(p)


class _NoCalibration:
    name = "none"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_NoCalibration":
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return p


# order is the tie-break: the simplest method wins a tie
_METHODS = {"none": _NoCalibration, "platt": _Platt, "isotonic": _Isotonic}


def _binary_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, _CLIP, 1 - _CLIP)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


@dataclass
class ThresholdCalibration:
    """Per-threshold calibrators for P(delay >= 15) and P(delay >= 30), then bands rebuilt."""

    method: str
    bands: Bands
    maps: dict[int, Any]
    tune_log_loss: dict[str, float]

    def transform(self, p: np.ndarray) -> np.ndarray:
        e = {t: exceedance(p, self.bands, t) for t in self.bands.lower_edges[1:]}
        for t, cal in self.maps.items():
            e[t] = cal.predict(e[t])
        return normalise_rows(rebuild_bands(e, self.bands))


def fit_threshold_calibration(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_tune: np.ndarray,
    y_tune: np.ndarray,
    dates_cal: np.ndarray,
    bands: Bands,
    same: bool,
) -> ThresholdCalibration:
    """Fit isotonic and Platt on the calibrate half, choose among them and none by log loss.

    `y_*` are band indices. Log loss is summed over the two thresholds and computed on the
    tuning half. When validation was not split (`same`), the choice is cross-validated over
    whole service dates of the one block instead, so a method is never scored on the rows it
    was fitted on. Ties go to the simpler method (none, then Platt).
    """
    obs = {t: (y_cal >= bands.lower_edges.index(t)).astype(float) for t in THRESHOLDS}
    obs_tune = {t: (y_tune >= bands.lower_edges.index(t)).astype(float) for t in THRESHOLDS}
    e_cal = {t: exceedance(p_cal, bands, t) for t in THRESHOLDS}
    e_tune = {t: exceedance(p_tune, bands, t) for t in THRESHOLDS}

    def fit_all(name: str, rows: np.ndarray | None = None) -> dict[int, Any]:
        sel = slice(None) if rows is None else rows
        return {t: _METHODS[name]().fit(e_cal[t][sel], obs[t][sel]) for t in THRESHOLDS}

    scores: dict[str, float] = {}
    if same:
        unique = np.unique(dates_cal)
        folds = min(5, len(unique))
        if folds >= 2:
            assignment = {d: i % folds for i, d in enumerate(unique)}
            fold_of = np.array([assignment[d] for d in dates_cal])
        for name in _METHODS:
            losses = []
            for f in range(folds if folds >= 2 else 0):
                held, rest = fold_of == f, fold_of != f
                if not held.any() or not rest.any():
                    continue
                maps = fit_all(name, rest)
                losses.append(
                    sum(_binary_log_loss(maps[t].predict(e_cal[t][held]), obs[t][held])
                        for t in THRESHOLDS)
                )  # fmt: skip
            scores[name] = float(np.mean(losses)) if losses else float("nan")
    else:
        for name in _METHODS:
            maps = fit_all(name)
            scores[name] = float(
                sum(_binary_log_loss(maps[t].predict(e_tune[t]), obs_tune[t]) for t in THRESHOLDS)
            )
    usable = {k: v for k, v in scores.items() if not np.isnan(v)}
    chosen = min(usable, key=lambda k: (usable[k], list(_METHODS).index(k))) if usable else "none"
    return ThresholdCalibration(chosen, bands, fit_all(chosen), scores)


@dataclass
class BandFit:
    design: Design
    booster: lgb.Booster
    bands: Bands
    calibrator: Any
    cv_log_loss: dict[str, float]
    best_iteration: int
    threshold: dict[int, ThresholdCalibration] | None = None  # by horizon bucket
    hedges: tuple[float, ...] = DEFAULT_BUCKET_EDGES

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(self.design.transform(df), num_iteration=self.best_iteration)
        return normalise_rows(np.asarray(raw))

    def predict_pre_threshold(self, df: pd.DataFrame) -> np.ndarray:
        """Band probabilities after band calibration, before the threshold recalibration."""
        return normalise_rows(self.calibrator.transform(self.predict_raw(df)))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_pre_threshold(df)
        if not self.threshold:
            return p
        out = p.copy()
        buckets = horizon_buckets(df["horizon_min"].to_numpy(dtype=float), self.hedges)
        for b in np.unique(buckets):
            if int(b) in self.threshold:
                idx = np.flatnonzero(buckets == b)
                out[idx] = self.threshold[int(b)].transform(p[idx])
        return out


def fit_bands(
    train: pd.DataFrame,
    tune: pd.DataFrame,
    calibrate: pd.DataFrame,
    bands: Bands,
    cfg: dict[str, Any],
    columns: list[str] | None = None,
) -> BandFit:
    hedges = tuple(
        float(e) for e in cfg.get("mondrian", {}).get("horizon_bucket_edges", DEFAULT_BUCKET_EDGES)
    )
    design = Design(columns).fit(train)
    y_train = band_index(train["label_delay_min"].to_numpy(dtype=float), bands)
    y_tune = band_index(tune["label_delay_min"].to_numpy(dtype=float), bands)
    params = {
        "objective": "multiclass",
        "num_class": bands.k,
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": cfg["seed"],
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": cfg["num_threads"],
        "lambda_l2": cfg["fixed"]["lambda_l2"],
        "max_depth": cfg["fixed"]["max_depth"],
        "bagging_fraction": cfg["fixed"]["bagging_fraction"],
        "bagging_freq": cfg["fixed"]["bagging_freq"],
        **cfg["band_model"],
    }
    train_set = lgb.Dataset(design.transform(train), y_train, categorical_feature="auto")
    valid_set = lgb.Dataset(design.transform(tune), y_tune, reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        cfg["max_rounds"],
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)],
    )
    best = max(1, booster.best_iteration)
    shell = BandFit(design, booster, bands, Identity(), {}, best, hedges=hedges)
    p_cal = shell.predict_raw(calibrate)
    y_cal = band_index(calibrate["label_delay_min"].to_numpy(dtype=float), bands)
    dates = calibrate["service_date"].to_numpy()
    calibrator, cv = choose_calibrator(p_cal, y_cal, dates)
    shell.calibrator, shell.cv_log_loss = calibrator, cv
    # threshold recalibration per horizon: fitted on the calibrate half, chosen on the tune half
    p_after = shell.predict_pre_threshold(calibrate)
    p_tune = shell.predict_pre_threshold(tune)
    y_tune_band = band_index(tune["label_delay_min"].to_numpy(dtype=float), bands)
    same = tune is calibrate
    h_cal = horizon_buckets(calibrate["horizon_min"].to_numpy(dtype=float), hedges)
    h_tune = horizon_buckets(tune["horizon_min"].to_numpy(dtype=float), hedges)
    shell.threshold = {}
    for h in sorted(set(h_cal.tolist())):
        mc, mt = h_cal == h, h_tune == h
        use_same = same or not mt.any()
        shell.threshold[int(h)] = fit_threshold_calibration(
            p_after[mc],
            y_cal[mc],
            p_after[mc] if use_same else p_tune[mt],
            y_cal[mc] if use_same else y_tune_band[mt],
            dates[mc],
            bands,
            same=use_same,
        )
    return shell
