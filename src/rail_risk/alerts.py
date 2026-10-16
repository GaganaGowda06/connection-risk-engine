"""Alert thresholds chosen by minimising expected cost under a stated cost matrix.

An alert fires when the probability is at or above the threshold. A false alert (fired, nothing
happened) costs `false_alert` and a missed alert (not fired, the event happened) costs
`missed_alert`. Both costs are assumptions held in config/alerts.yaml. The threshold is chosen
on validation pairs only and then scored once on test pairs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

NEVER = 1.01  # a threshold above every probability: no alert is ever sent


@dataclass(frozen=True)
class AlertConfig:
    false_alert: float
    missed_alert: float
    sensitivity_missed_alert: tuple[float, ...]
    grid_step: float
    tau: float | None = None  # the chosen threshold for the likely-to-miss alert


def load_alert_config(path: Path | str = "config/alerts.yaml") -> AlertConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = AlertConfig(
        false_alert=float(raw["costs"]["false_alert"]),
        missed_alert=float(raw["costs"]["missed_alert"]),
        sensitivity_missed_alert=tuple(float(x) for x in raw.get("sensitivity_missed_alert", ())),
        grid_step=float(raw.get("grid_step", 0.01)),
        tau=None if "chosen" not in raw else float(raw["chosen"]["tau"]),
    )
    if min(cfg.false_alert, cfg.missed_alert) <= 0 or not 0 < cfg.grid_step < 1:
        raise ValueError("alert costs must be positive and grid_step between 0 and 1")
    return cfg


def threshold_grid(step: float) -> np.ndarray:
    """0 (always alert) up to 1 in `step`s, then NEVER."""
    return np.append(np.round(np.arange(0.0, 1.0 + step / 2, step), 10), NEVER)


def policy_stats(
    p: np.ndarray, y: np.ndarray, tau: float, false_alert: float, missed_alert: float
) -> dict[str, Any]:
    """Counts, precision, recall, alert rate and expected cost per pair for one threshold."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float) > 0.5
    fire = p >= tau
    tp, fp, fn = int((fire & y).sum()), int((fire & ~y).sum()), int((~fire & y).sum())
    n = len(p)
    return {
        "tau": float(tau),
        "n": n,
        "positives": int(y.sum()),
        "alerts": int(fire.sum()),
        "true_alerts": tp,
        "false_alerts": fp,
        "missed_alerts": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "alert_rate": float(fire.mean()) if n else None,
        "cost_per_pair": (false_alert * fp + missed_alert * fn) / n if n else None,
    }


def choose_threshold(
    p: np.ndarray, y: np.ndarray, false_alert: float, missed_alert: float, step: float = 0.01
) -> float:
    """The threshold with the lowest expected cost on these rows. A tie goes to the higher one."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float) > 0.5
    grid = threshold_grid(step)
    costs = np.array(
        [
            (false_alert * (p >= t)[~y].sum() + missed_alert * (~(p >= t))[y].sum()) / len(p)
            for t in grid
        ]
    )
    best = np.flatnonzero(costs <= costs.min() + 1e-12)
    return float(grid[best[-1]])


def trivial_policies(
    p: np.ndarray, y: np.ndarray, false_alert: float, missed_alert: float
) -> dict[str, dict[str, Any]]:
    return {
        "always_alert": policy_stats(p, y, 0.0, false_alert, missed_alert),
        "never_alert": policy_stats(p, y, NEVER, false_alert, missed_alert),
    }


def alert_report(
    name: str,
    validation: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    cfg: AlertConfig,
) -> dict[str, Any]:
    """Choose the threshold on validation, score it on test, and show how it moves with the
    missed-alert cost. The test rows are never used to pick any threshold."""
    pv, yv = validation
    pt, yt = test
    tau = choose_threshold(pv, yv, cfg.false_alert, cfg.missed_alert, cfg.grid_step)
    sensitivity = {}
    for missed in (cfg.missed_alert, *cfg.sensitivity_missed_alert):
        t = choose_threshold(pv, yv, cfg.false_alert, missed, cfg.grid_step)
        sensitivity[f"{missed:g}"] = {
            "tau": t,
            "theoretical_tau_if_calibrated": cfg.false_alert / (cfg.false_alert + missed),
            "validation_alert_rate": float((np.asarray(pv) >= t).mean()),
        }
    return {
        "name": name,
        "costs": {"false_alert": cfg.false_alert, "missed_alert": cfg.missed_alert},
        "tau": tau,
        "validation": {
            "chosen": policy_stats(pv, yv, tau, cfg.false_alert, cfg.missed_alert),
            **trivial_policies(pv, yv, cfg.false_alert, cfg.missed_alert),
        },
        "test": {
            "chosen": policy_stats(pt, yt, tau, cfg.false_alert, cfg.missed_alert),
            **trivial_policies(pt, yt, cfg.false_alert, cfg.missed_alert),
        },
        "sensitivity_to_missed_alert_cost": sensitivity,
    }
