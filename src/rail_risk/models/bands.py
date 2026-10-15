"""LightGBM multiclass delay-band model with a validation-chosen probability calibration."""

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from rail_risk.evaluation import Bands, band_index
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


@dataclass
class BandFit:
    design: Design
    booster: lgb.Booster
    bands: Bands
    calibrator: Any
    cv_log_loss: dict[str, float]
    best_iteration: int

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(self.design.transform(df), num_iteration=self.best_iteration)
        return normalise_rows(np.asarray(raw))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return normalise_rows(self.calibrator.transform(self.predict_raw(df)))


def fit_bands(
    train: pd.DataFrame,
    tune: pd.DataFrame,
    calibrate: pd.DataFrame,
    bands: Bands,
    cfg: dict[str, Any],
    columns: list[str] | None = None,
) -> BandFit:
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
    shell = BandFit(design, booster, bands, Identity(), {}, best)
    p_cal = shell.predict_raw(calibrate)
    y_cal = band_index(calibrate["label_delay_min"].to_numpy(dtype=float), bands)
    dates = calibrate["service_date"].to_numpy()
    calibrator, cv = choose_calibrator(p_cal, y_cal, dates)
    shell.calibrator, shell.cv_log_loss = calibrator, cv
    return shell
