"""Recalibration of the simulated P(miss), fitted on validation pairs only.

The simulator's P(miss) over-predicts. A map from the simulated probability to the observed miss
frequency is chosen among none, Platt scaling and isotonic regression. Validation pairs are split
by date: the methods are fitted on the later half (calibrate) and scored by log loss on the
earlier half (tune), the same scheme as the band threshold recalibration. Ties go to the simpler
method. The map is applied only to forecasts whose inbound has not arrived, because once it has
arrived the delay is known and the simulation is not the thing being corrected.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rail_risk.models.bands import _METHODS, _binary_log_loss

CLIP = 1e-6


@dataclass
class PMissCalibration:
    method: str  # "none", "platt" or "isotonic"
    params: dict[str, Any] = field(default_factory=dict)
    tune_log_loss: dict[str, float] = field(default_factory=dict)
    n_tune: int = 0
    n_calibrate: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def apply(self, p: np.ndarray | float) -> np.ndarray:
        x = np.atleast_1d(np.asarray(p, dtype=float))
        if self.method == "none":
            return x
        if self.method == "platt":
            if "constant" in self.params:
                return np.full_like(x, self.params["constant"])
            q = np.clip(x, CLIP, 1 - CLIP)
            z = self.params["coef"] * np.log(q / (1 - q)) + self.params["intercept"]
            return 1.0 / (1.0 + np.exp(-z))
        return np.interp(x, self.params["x"], self.params["y"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "tune_log_loss": self.tune_log_loss,
            "n_tune": self.n_tune,
            "n_calibrate": self.n_calibrate,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PMissCalibration":
        return cls(
            raw["method"], raw.get("params", {}), raw.get("tune_log_loss", {}),
            raw.get("n_tune", 0), raw.get("n_calibrate", 0), raw.get("provenance", {}),
        )  # fmt: skip


def _params(name: str, fitted: Any) -> dict[str, Any]:
    if name == "none":
        return {}
    if name == "platt":
        if fitted.constant is not None:
            return {"constant": float(fitted.constant)}
        return {
            "coef": float(fitted.model.coef_[0, 0]),
            "intercept": float(fitted.model.intercept_[0]),
        }
    iso = fitted.model
    return {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}


def fit_pmiss_calibration(
    p: np.ndarray, y: np.ndarray, dates: np.ndarray, provenance: dict[str, Any] | None = None
) -> PMissCalibration:
    """Choose and fit the map. `dates` split the rows: the earlier half tunes, the later half
    calibrates. With fewer than two dates the method is none."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    udates = np.array(sorted(set(dates.tolist())))
    if len(udates) < 2 or len(p) < 40:
        return PMissCalibration("none", provenance=provenance or {})
    first = set(udates[: len(udates) // 2].tolist())
    tune = np.array([d in first for d in dates.tolist()])
    cal = ~tune
    scores, fits = {}, {}
    for name, cls in _METHODS.items():
        fitted = cls().fit(p[cal], y[cal])
        fits[name] = fitted
        scores[name] = _binary_log_loss(fitted.predict(p[tune]), y[tune])
    best = min(scores, key=lambda k: (scores[k], list(_METHODS).index(k)))
    return PMissCalibration(
        best, _params(best, fits[best]), scores, int(tune.sum()), int(cal.sum()), provenance or {}
    )


def split_frame(pred: pd.DataFrame) -> pd.DataFrame:
    """Validation rows with a known outcome whose inbound had not arrived: the fitting set."""
    return pred[
        (pred["block"] == "validation") & pred["missed"].notna() & (pred["state"] != "arrived")
    ]


def apply_to_predictions(pred: pd.DataFrame, cal: PMissCalibration) -> pd.DataFrame:
    """Adds p_miss_cal: the map for not-yet-arrived forecasts, the raw value for arrived ones."""
    out = pred.copy()
    mapped = cal.apply(out["p_miss"].to_numpy())
    out["p_miss_cal"] = np.where(out["state"] == "arrived", out["p_miss"], mapped)
    return out


def save(cal: PMissCalibration, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal.to_dict(), indent=2) + "\n", encoding="utf-8")


def load(path: Path) -> PMissCalibration:
    return PMissCalibration.from_dict(json.loads(path.read_text(encoding="utf-8")))
