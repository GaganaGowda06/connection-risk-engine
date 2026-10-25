"""P(miss) options for a connection: physical, LightGBM classifier, logistic regression.

Everything here is route, station, hub and operator agnostic: the feature names carry none of
those words (a test enforces it). A row is a pair at one horizon; miss = inbound arrival lateness
A minus outbound departure lateness D is greater than the planned slack (gap minus MCT).

  physical (a)  A and D are distributions from LightGBM quantile heads (levels LEVELS), widened by
                a conformal scale from the out-of-group pool. A is a point once the inbound has an
                actual arrival by T, D is a point once the outbound has departed by T. P(miss) =
                p_cancel + (1 - p_cancel) * P(A - D > slack), independent A and D, on a grid.
  lgbm (b)      binary LightGBM on the state features plus (a)'s probability.
  logistic (c)  per-horizon logistic regression on slack, the inbound's lateness and flags.
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
GRID = 20
EPS_WIDTH = 0.5
BANNED = ("route", "rid", "tpl", "tiploc", "crs", "toc", "operator", "hub", "corr")

COMMON = ["horizon_min", "dep_hour", "dow", "weekend", "cong_recent_n", "cong_recent_late_min",
          "cong_cancel_n", "cong_sched_n"]  # fmt: skip
IN_STATE = ["in_through", "in_arrived", "in_arr_late_min", "in_fc_late_min", "in_last_late_min",
            "in_last_age_min", "in_cancel_known", "in_eff_late_min", "in_src",
            "mins_to_plan_arr"]  # fmt: skip
OUT_STATE = ["out_through", "out_departed", "out_dep_late_min", "out_fc_late_min",
             "out_arr_late_min", "out_arr_actual", "out_last_late_min", "out_last_age_min",
             "out_cancel_known", "out_eff_late_min"]  # fmt: skip
PAIR = ["slack_min", "mct_min", "proj_short_min"]
FEATS_A = [*COMMON, *IN_STATE]
FEATS_D = [*COMMON, *OUT_STATE]
FEATS_B = [*COMMON, *IN_STATE, *OUT_STATE, *PAIR]
FEATS_C = ["slack_min", "in_eff_late_min", "in_arrived", "in_cancel_known"]


def check_features(names: list[str]) -> None:
    bad = [n for n in names if any(b in n.lower() for b in BANNED)]
    if bad:
        raise ValueError(f"identity columns reached the model: {bad}")


for _f in (FEATS_A, FEATS_D, FEATS_B, FEATS_C):
    check_features(_f)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """Model features (float32) and labels from a pair table, nothing beyond the T-state columns."""
    m = 1.0 / 60.0
    o = pd.DataFrame(index=df.index)
    o["horizon_min"] = df["horizon_min"].astype("float32")
    t = df["plan_dep_out"].dt.tz_convert("Europe/London")
    o["dep_hour"] = (t.dt.hour + t.dt.minute / 60.0).astype("float32")
    o["dow"] = t.dt.dayofweek.astype("float32")
    o["weekend"] = (t.dt.dayofweek >= 5).astype("float32")
    for c in ("cong_recent_n", "cong_recent_late_min", "cong_cancel_n", "cong_sched_n"):
        o[c] = df[c].astype("float32")
    o["in_through"] = df["in_through"].astype("float32")
    o["in_arrived"] = df["in_arr_late_s"].notna().astype("float32")
    o["in_arr_late_min"] = (df["in_arr_late_s"] * m).astype("float32")
    o["in_fc_late_min"] = (df["in_fc_late_s"] * m).astype("float32")
    o["in_last_late_min"] = (df["in_last_late_s"] * m).astype("float32")
    o["in_last_age_min"] = (df["in_last_age_s"] * m).astype("float32")
    o["in_cancel_known"] = df["in_cancel_known"].astype("float32")
    src = np.where(df["in_arr_late_s"].notna(), 3, np.where(df["in_fc_late_s"].notna(), 2,
                   np.where(df["in_last_late_s"].notna(), 1, 0)))  # fmt: skip
    eff = df["in_arr_late_s"].fillna(df["in_fc_late_s"]).fillna(df["in_last_late_s"]).fillna(0.0)
    o["in_src"] = src.astype("float32")
    o["in_eff_late_min"] = (eff * m).astype("float32")
    o["mins_to_plan_arr"] = (df["horizon_min"] + df["gap"]).astype("float32")
    o["out_through"] = df["out_through"].astype("float32")
    o["out_departed"] = df["out_dep_late_s"].notna().astype("float32")
    o["out_dep_late_min"] = (df["out_dep_late_s"] * m).astype("float32")
    o["out_fc_late_min"] = (df["out_fc_late_s"] * m).astype("float32")
    o["out_arr_late_min"] = (df["out_arr_late_s"] * m).astype("float32")
    o["out_arr_actual"] = df["out_arr_actual"].astype("float32")
    o["out_last_late_min"] = (df["out_last_late_s"] * m).astype("float32")
    o["out_last_age_min"] = (df["out_last_age_s"] * m).astype("float32")
    o["out_cancel_known"] = df["out_cancel_known"].astype("float32")
    oeff = (
        df["out_dep_late_s"].fillna(df["out_fc_late_s"]).fillna(df["out_arr_late_s"])
        .fillna(df["out_last_late_s"]).fillna(0.0)
    )  # fmt: skip
    o["out_eff_late_min"] = (oeff * m).astype("float32")
    o["slack_min"] = df["slack_min"].astype("float32")
    o["mct_min"] = df["mct_min"].astype("float32")
    o["proj_short_min"] = (o["in_eff_late_min"] - o["out_eff_late_min"] - o["slack_min"]).astype(
        "float32"
    )
    return o


# quantile heads and the conformal scale ----------------------------------------------------------
QPARAMS = {"num_leaves": 31, "learning_rate": 0.1, "min_data_in_leaf": 200, "feature_fraction": 0.9,
           "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8,
           "seed": 20261015}  # fmt: skip
QROUNDS = 150


@dataclass
class QuantileSet:
    feats: list[str]
    boosters: dict = field(default_factory=dict)
    scale: float = 0.0  # conformal CQR scale for the 10 to 90 range

    def fit(self, x: pd.DataFrame, y: np.ndarray, max_rows: int = 400_000, seed: int = 0):
        if len(x) > max_rows:
            ix = np.random.default_rng(seed).choice(len(x), max_rows, replace=False)
            x, y = x.iloc[ix], y[ix]
        for q in LEVELS:
            p = {**QPARAMS, "objective": "quantile", "alpha": q}
            self.boosters[q] = lgb.train(p, lgb.Dataset(x[self.feats], y), QROUNDS)
        return self

    def raw(self, x: pd.DataFrame) -> np.ndarray:
        q = np.column_stack([self.boosters[q].predict(x[self.feats]) for q in LEVELS])
        return np.sort(q, axis=1)

    def calibrate(self, x: pd.DataFrame, y: np.ndarray) -> "QuantileSet":
        q = self.raw(x)
        lo, hi = q[:, 1], q[:, 5]
        w = np.maximum(hi - lo, EPS_WIDTH)
        s = np.maximum(lo - y, y - hi) / w
        n = len(s)
        self.scale = float(np.quantile(s, min(1.0, np.ceil((n + 1) * 0.8) / n)))
        return self

    def quantile_grid(self, x: pd.DataFrame, n: int = GRID) -> np.ndarray:
        """Calibrated quantile function at taus (i + 0.5) / n, shape (rows, n)."""
        q = self.raw(x)
        w = np.maximum(q[:, 5] - q[:, 1], EPS_WIDTH) * self.scale
        taus = (np.arange(n) + 0.5) / n
        lv = np.array(LEVELS)
        out = np.empty((len(q), n))
        shift = np.interp(taus, [0.0, 0.1, 0.5, 0.9, 1.0], [-1.0, -1.0, 0.0, 1.0, 1.0])
        for j, t in enumerate(taus):
            k = int(np.clip(np.searchsorted(lv, t) - 1, 0, len(lv) - 2))
            slope = (q[:, k + 1] - q[:, k]) / (lv[k + 1] - lv[k])
            out[:, j] = q[:, k] + slope * (t - lv[k]) + shift[j] * w
        return np.maximum.accumulate(out, axis=1)


def physical_p(x: pd.DataFrame, a: QuantileSet, d: QuantileSet, p_cancel: dict) -> np.ndarray:
    """P(miss) from the two calibrated distributions. x is prep() output."""
    n = len(x)
    ga = a.quantile_grid(x)
    gd = d.quantile_grid(x)
    arrived = x["in_arrived"].to_numpy() > 0
    departed = x["out_departed"].to_numpy() > 0
    ga[arrived] = x["in_arr_late_min"].to_numpy()[arrived][:, None]
    gd[departed] = x["out_dep_late_min"].to_numpy()[departed][:, None]
    slack = x["slack_min"].to_numpy()
    out = np.empty(n)
    for s in range(0, n, 40_000):
        sl = slice(s, s + 40_000)
        diff = ga[sl, :, None] - gd[sl, None, :]
        out[sl] = (diff > slack[sl, None, None]).mean(axis=(1, 2))
    h = x["horizon_min"].to_numpy().astype(int)
    pc = np.array([p_cancel.get(int(k), 0.0) for k in h])
    pc = np.where(arrived, 0.0, pc)
    p = pc + (1 - pc) * out
    return np.where(x["in_cancel_known"].to_numpy() > 0, 1.0, p)


def fit_p_cancel(df: pd.DataFrame) -> dict:
    """Share of not-yet-arrived inbound services that never reached the hub, per horizon."""
    nya = df["in_arr_late_s"].isna() & ~df["in_cancel_known"].astype(bool)
    d = df[nya]
    return {int(h): float(g["act_arr_in"].isna().mean()) for h, g in d.groupby("horizon_min")}


# classifier and logistic ---------------------------------------------------------------------
CPARAMS = {"objective": "binary", "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 500,
           "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
           "verbose": -1, "num_threads": 8, "seed": 20261015}  # fmt: skip
CROUNDS = 300


def fit_classifier(x: pd.DataFrame, y: np.ndarray, params: dict | None = None, rounds=CROUNDS):
    p = {**CPARAMS, **(params or {})}
    return lgb.train(p, lgb.Dataset(x, y), rounds)


class Logistic:
    """One logistic regression per horizon on FEATS_C (standardised)."""

    def __init__(self):
        self.models: dict = {}

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        h = x["horizon_min"].to_numpy().astype(int)
        for k in np.unique(h):
            m = h == k
            z = x.loc[m, FEATS_C].fillna(0.0)
            mu, sd = z.mean(), z.std().replace(0, 1.0)
            lr = LogisticRegression(C=1.0, max_iter=500).fit((z - mu) / sd, y[m])
            self.models[int(k)] = (lr, mu, sd)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(x))
        h = x["horizon_min"].to_numpy().astype(int)
        for k, (lr, mu, sd) in self.models.items():
            m = h == k
            if m.any():
                z = x.loc[m, FEATS_C].fillna(0.0)
                out[m] = lr.predict_proba((z - mu) / sd)[:, 1]
        return out


def fit_isotonic(p: np.ndarray, y: np.ndarray, horizons: np.ndarray) -> dict:
    return {
        int(k): IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            p[horizons == k], y[horizons == k]
        )
        for k in np.unique(horizons)
    }


def apply_isotonic(iso: dict, p: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    out = p.copy()
    for k, m in iso.items():
        sel = horizons == k
        if sel.any():
            out[sel] = m.predict(p[sel])
    return out
