"""The serving bundle: everything the API needs from a training run, in one artefact.

Registered with MLflow as the model "connection-risk". The service loads it by alias, so a later
retrain only has to move the alias. The bundle is a pickle written by this project's own
training run into its own store; it is not a format to load from an untrusted source.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from rail_risk.evaluation import Bands, HistoricalBaseline
from rail_risk.models.bands import BandFit
from rail_risk.models.quantile import QuantileModel

BUNDLE_FILE = "serving_bundle.pkl"
BASELINE_FILE = "baseline.pkl"


@dataclass
class ModelBundle:
    quantile: QuantileModel
    bands: BandFit
    band_def: Bands
    horizons: tuple[int, ...] = (120, 60, 15)
    continuous_horizon: bool = False  # True once the model takes minutes_to_arrival as a feature
    meta: dict[str, Any] = field(default_factory=dict)

    def model_horizon(self, minutes: float) -> tuple[int, bool]:
        """(horizon to give the model, whether it was snapped from the real minutes).

        A model with a continuous horizon takes the minutes as they are. One trained on a few
        fixed horizons gets the nearest, and the response says so.
        """
        if self.continuous_horizon:
            return int(round(minutes)), False
        nearest = min(self.horizons, key=lambda h: abs(h - minutes))
        return nearest, abs(nearest - minutes) > 0.5


def save_bundle(bundle: ModelBundle, baseline: HistoricalBaseline | None, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, folder / BUNDLE_FILE)
    if baseline is not None:
        joblib.dump(baseline, folder / BASELINE_FILE)


def load_bundle(folder: Path) -> ModelBundle:
    return joblib.load(folder / BUNDLE_FILE)  # noqa: S301 - our own artefact


def load_baseline(folder: Path) -> HistoricalBaseline | None:
    path = folder / BASELINE_FILE
    return joblib.load(path) if path.exists() else None  # noqa: S301 - our own artefact
