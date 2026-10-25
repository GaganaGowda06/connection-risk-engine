"""Range calibration for unseen routes: three methods, all fitted on out-of-route residuals.

The pool is made of (a) predictions of the leave-route-group-out tuning-fold models on the routes
each fold did not train on, over the train block, and (b) validation rows of the training routes.
No held-out route is in the pool. Groups use only numbers computable for any route: horizon,
lateness band at T, scheduled journey length and public stops left. No route, station or operator.

  cross     residual quantiles (10% and 90%) of the median forecast, per horizon
  cqr       conformalised quantile regression: the model's own 10-90 range, widened by a quantile
            of scores scaled by the range width (difficulty-scaled), per horizon
  mondrian  residual quantiles per (horizon, lateness band, length class, stops left) cell, with a
            fall-back to coarser cells when a cell has fewer than min_cell pool rows
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NOMINAL = 0.8
LOW_Q, HIGH_Q = 0.1, 0.9
MIN_CELL = 150
EPS_WIDTH = 0.5
LATE_BANDS = [-np.inf, 0.0, 5.0, 15.0, np.inf]  # none or early, 0-5, 5-15, 15 and over
LENGTH_EDGES = [40.0, 100.0]  # scheduled minutes: short, medium, long
STOPS_EDGES = [3, 7, 16]  # public stops left: 0-2, 3-6, 7-15, 16 and over
LEVELS = [("h", "lb", "lc", "sb"), ("h", "lb", "lc"), ("h", "lb"), ("h",)]
METHODS = ("cross", "cqr", "mondrian")
MIN_POOL_PER_HORIZON = 100  # fewer calibration rows than this at a horizon is an error


def group_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The transferable groups of each row, as small integers."""
    late = df["late_now_min"].fillna(0.0).to_numpy()
    out = pd.DataFrame(index=df.index)
    out["h"] = df["horizon_min"].to_numpy().astype(int)
    out["lb"] = np.digitize(late, LATE_BANDS[1:-1], right=True)
    out["lc"] = np.digitize(df["sched_duration_min"].to_numpy(), LENGTH_EDGES)
    out["sb"] = np.digitize(df["stops_remaining"].fillna(0).to_numpy(), STOPS_EDGES)
    return out


@dataclass
class RangeCalibrator:
    method: str
    offsets: dict = field(default_factory=dict)
    scale: dict = field(default_factory=dict)
    fallback_rows: int = 0
    min_cell: int = MIN_CELL

    def fit(self, pool: pd.DataFrame) -> "RangeCalibrator":
        """pool: point, lo, hi, target_min and the group columns, from out-of-route predictions."""
        res = pool["target_min"] - pool["point"]
        g = group_frame(pool)
        small = g["h"].value_counts()
        small = small[small < MIN_POOL_PER_HORIZON]
        if len(small) or pool.empty:
            raise ValueError(
                f"too few calibration rows per horizon (minimum {MIN_POOL_PER_HORIZON}): "
                f"{dict(small) or 'empty pool'}"
            )
        if self.method == "cross":
            for h, r in res.groupby(g["h"]):
                self.offsets[(int(h),)] = (
                    float(r.quantile(LOW_Q)),
                    float(r.quantile(HIGH_Q)),
                    len(r),
                )
        elif self.method == "cqr":
            w = np.maximum(pool["hi"] - pool["lo"], EPS_WIDTH)
            s = np.maximum(pool["lo"] - pool["target_min"], pool["target_min"] - pool["hi"]) / w
            for h, v in s.groupby(g["h"]):
                n = len(v)
                level = min(1.0, np.ceil((n + 1) * NOMINAL) / n)
                self.scale[int(h)] = float(v.quantile(level))
        elif self.method == "mondrian":
            for level in LEVELS:
                for key, r in res.groupby([g[c] for c in level]):
                    key = key if isinstance(key, tuple) else (key,)
                    if len(r) >= self.min_cell:
                        self.offsets[(level, tuple(int(k) for k in key))] = (
                            float(r.quantile(LOW_Q)),
                            float(r.quantile(HIGH_Q)),
                            len(r),
                        )
        else:
            raise ValueError(f"unknown method {self.method!r}")
        return self

    def apply(self, df: pd.DataFrame, point: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        g = group_frame(df)
        if self.method == "cross":
            off = [self.offsets.get((int(h),), (0.0, 0.0, 0)) for h in g["h"]]
            lo_n = point + np.array([o[0] for o in off])
            hi_n = point + np.array([o[1] for o in off])
        elif self.method == "cqr":
            q = np.array([self.scale.get(int(h), 0.0) for h in g["h"]])
            w = np.maximum(hi - lo, EPS_WIDTH)
            lo_n, hi_n = lo - q * w, hi + q * w
        else:
            lo_off = np.zeros(len(df))
            hi_off = np.zeros(len(df))
            done = np.zeros(len(df), bool)
            fine = 0
            for i, level in enumerate(LEVELS):
                keys = list(zip(*(g[c].to_numpy() for c in level), strict=True))
                for j, k in enumerate(keys):
                    if done[j]:
                        continue
                    hit = self.offsets.get((level, tuple(int(x) for x in k)))
                    if hit is not None:
                        lo_off[j], hi_off[j], done[j] = hit[0], hit[1], True
                        fine += i > 0
            self.fallback_rows = int(fine)
            lo_n, hi_n = point + lo_off, point + hi_off
        lo_n, hi_n = np.minimum(lo_n, hi_n), np.maximum(lo_n, hi_n)
        return lo_n, hi_n


def coverage_table(
    frame: pd.DataFrame, lo: np.ndarray, hi: np.ndarray, horizons: tuple[int, ...] = (15, 60)
) -> dict:
    """Coverage and mean width pooled, per horizon, and per route (worst route per horizon)."""
    y = frame["target_min"].to_numpy()
    cov = ((lo <= y) & (y <= hi)).astype(float)
    wid = hi - lo
    out: dict = {"overall": float(cov.mean()), "mean_width": float(wid.mean())}
    h = frame["horizon_min"].to_numpy()
    r = frame["route"].to_numpy()
    for k in horizons:
        m = h == k
        per = pd.Series(cov[m]).groupby(r[m]).mean()
        out[k] = {
            "coverage": float(cov[m].mean()),
            "mean_width": float(wid[m].mean()),
            "worst_route": per.idxmin(),
            "worst_route_coverage": float(per.min()),
            "routes": {str(a): float(b) for a, b in per.items()},
        }
    return out


@dataclass
class Calibrated:
    """A fitted Booster3 plus the out-of-route range calibration and the P(late) calibrators."""

    model: object  # train.Booster3
    ranges: RangeCalibrator
    p_late: dict  # horizon -> IsotonicRegression fitted on out-of-route raw scores
    meta: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
        raw = self.model.predict(df, base)
        lo, hi = self.ranges.apply(
            df, raw["point"].to_numpy(), raw["lo"].to_numpy(), raw["hi"].to_numpy()
        )
        h = df["horizon_min"].to_numpy()
        p = raw["p_late_raw"].to_numpy()
        cal = np.array(
            [self.p_late[int(k)].predict([v])[0] if int(k) in self.p_late else v
             for k, v in zip(h, p, strict=True)]
        )  # fmt: skip
        out = pd.DataFrame(
            {
                "point": raw["point"].to_numpy(), "lo": lo, "hi": hi,
                "raw_lo": raw["lo"].to_numpy(), "raw_hi": raw["hi"].to_numpy(),
                "p_late": np.clip(cal, 0.0, 1.0), "p_late_stage3": raw["p_late"].to_numpy(),
                "p_late_raw": p,
            },
            index=df.index,
        )  # fmt: skip
        return out

    def save(self, path) -> None:
        import joblib

        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model")
        joblib.dump(
            {"ranges": self.ranges, "p_late": self.p_late, "meta": self.meta},
            path / "calibration.joblib",
        )

    @staticmethod
    def load(path) -> "Calibrated":
        import joblib

        from rail_risk.network_model import train as T

        meta = joblib.load(path / "calibration.joblib")
        return Calibrated(
            T.Booster3.load(path / "model"), meta["ranges"], meta["p_late"], meta["meta"]
        )
