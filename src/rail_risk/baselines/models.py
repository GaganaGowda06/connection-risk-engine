"""The fixed baselines that tests 2 to 5 must beat. All are fitted on training dates only.

persistence      lateness at T carried to arrival (no actual yet means on time); the 80% range adds
                 the training residual quantiles of that rule per horizon; P(late >= 15) is the
                 training late rate inside bins of the persistence lateness, per horizon.
historical       the lateness distribution of the same route and arrival hour (falling back to the
                 route, then to the network, when a cell has fewer than MIN_CELL services):
                 median, 10% and 90% quantiles, and the late share. It ignores the state at T.
darwin_forecast  reference only, not a GOAL baseline: Darwin's own latest forecast at T, with the
                 same residual-quantile range and the same binned late rate.

Held-out routes get a historical cell from their own earlier dates. That is deliberate: it makes
the bar harder than a route the baseline has never seen.
Lateness is in minutes in this module (the snapshot table stores seconds).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LATE_MIN = 15.0
MIN_CELL = 30
LOW_Q, HIGH_Q = 0.10, 0.90
BIN_EDGES = np.array([-np.inf, 0.0, 2.0, 5.0, 10.0, 15.0, 30.0, np.inf])
METHODS = ("persistence", "historical", "darwin_forecast")


@dataclass
class Fitted:
    horizons: list[int]
    resid_q: dict[tuple[str, int], tuple[float, float]] = field(default_factory=dict)
    late_by_bin: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    hist_cell: dict[tuple[str, int], tuple[float, float, float, float, int]] = field(
        default_factory=dict
    )
    hist_route: dict[str, tuple[float, float, float, float, int]] = field(default_factory=dict)
    hist_all: tuple[float, float, float, float, int] = (0.0, 0.0, 0.0, 0.0, 0)
    fit_rows: int = 0
    fit_services: int = 0


def _minutes(s: pd.Series) -> np.ndarray:
    return s.astype("float64").to_numpy() / 60.0


def point_forecasts(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Point forecasts of arrival lateness in minutes for the state-based methods."""
    last = df["last_actual_late_s"].astype("float64").to_numpy() / 60.0
    persistence = np.where(np.isnan(last), 0.0, last)
    fc = df["fc_late_s"].astype("float64").to_numpy() / 60.0
    forecast = np.where(np.isnan(fc), persistence, fc)
    return {"persistence": persistence, "darwin_forecast": forecast}


def _stats(x: np.ndarray) -> tuple[float, float, float, float, int]:
    n = len(x)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0)
    q = np.quantile(x, [LOW_Q, 0.5, HIGH_Q])
    return (float(q[0]), float(q[1]), float(q[2]), (float((x >= LATE_MIN).sum()) + 1) / (n + 2), n)


def fit(train: pd.DataFrame) -> Fitted:
    """Fit every baseline on training-date rows that have a target and a known schedule."""
    t = train[train["has_target"] & train["schedule_known"]].copy()
    horizons = sorted(t["horizon_min"].unique().tolist())
    fitted = Fitted(horizons=horizons, fit_rows=len(t), fit_services=t["rid"].nunique())
    y = _minutes(t["target_late_s"])
    points = point_forecasts(t)
    for h in horizons:
        m = (t["horizon_min"] == h).to_numpy()
        for name in ("persistence", "darwin_forecast"):
            resid = y[m] - points[name][m]
            q = np.quantile(resid, [LOW_Q, HIGH_Q]) if m.any() else np.array([0.0, 0.0])
            fitted.resid_q[(name, h)] = (float(q[0]), float(q[1]))
            bins = np.digitize(points[name][m], BIN_EDGES[1:-1])
            rate = np.full(len(BIN_EDGES) - 1, np.nan)
            late = (y[m] >= LATE_MIN).astype(float)
            for b in range(len(rate)):
                sel = bins == b
                if sel.any():
                    rate[b] = (late[sel].sum() + 1) / (sel.sum() + 2)
            overall = (late.sum() + 1) / (len(late) + 2) if len(late) else 0.5
            fitted.late_by_bin[(name, h)] = np.where(np.isnan(rate), overall, rate)
    # one row per service for the historical cells, so a horizon does not weight a service
    svc = t.sort_values("horizon_min").drop_duplicates(["rid", "route"])
    ys = _minutes(svc["target_late_s"])
    fitted.hist_all = _stats(ys)
    for route, g in svc.groupby("route"):
        fitted.hist_route[route] = _stats(_minutes(g["target_late_s"]))
    for (route, hour), g in svc.groupby(["route", "arr_hour_local"]):
        fitted.hist_cell[(route, int(hour))] = _stats(_minutes(g["target_late_s"]))
    return fitted


def _hist_lookup(f: Fitted, route: str, hour: int) -> tuple[float, float, float, float, int]:
    cell = f.hist_cell.get((route, int(hour)))
    if cell and cell[4] >= MIN_CELL:
        return cell
    r = f.hist_route.get(route)
    if r and r[4] >= MIN_CELL:
        return r
    return f.hist_all


def predict(f: Fitted, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Point, 80% range and P(late >= 15) for each method, aligned to df's rows."""
    out: dict[str, pd.DataFrame] = {}
    points = point_forecasts(df)
    horizon = df["horizon_min"].to_numpy()
    for name in ("persistence", "darwin_forecast"):
        pt = points[name]
        lo_off = np.array([f.resid_q.get((name, h), (0.0, 0.0))[0] for h in horizon])
        hi_off = np.array([f.resid_q.get((name, h), (0.0, 0.0))[1] for h in horizon])
        bins = np.digitize(pt, BIN_EDGES[1:-1])
        p = np.array(
            [
                f.late_by_bin[(name, h)][b] if (name, h) in f.late_by_bin else 0.5
                for h, b in zip(horizon, bins, strict=True)
            ]
        )
        out[name] = pd.DataFrame({"point": pt, "lo": pt + lo_off, "hi": pt + hi_off, "p_late": p})
    cells = [
        _hist_lookup(f, r, h)
        for r, h in zip(df["route"].to_numpy(), df["arr_hour_local"].to_numpy(), strict=True)
    ]
    arr = np.array([c[:4] for c in cells], dtype=float).reshape(-1, 4)
    out["historical"] = pd.DataFrame(
        {"point": arr[:, 1], "lo": arr[:, 0], "hi": arr[:, 2], "p_late": arr[:, 3]}
    )
    for v in out.values():
        v.index = df.index
    return out
