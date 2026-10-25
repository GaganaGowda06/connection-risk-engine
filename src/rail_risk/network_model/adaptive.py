"""Online adaptive calibration of the 80% range, row by row in time order.

Every method starts from the global CQR calibration (scores from the out-of-route pool) and has
no history for a route or group at the start of a block. A row scored at its snapshot time T may
use the outcome of another row only if that row's train had arrived (actual arrival time) at or
before T. Nothing else crosses time: the engine processes a stream in order of T and keeps pending
outcomes in a heap until their arrival time passes.

  aci_route    Adaptive Conformal Inference (Gibbs and Candes 2021) with one miscoverage level per
               (route, horizon) stream: alpha <- alpha + gamma * (target - miss)
  aci_group    the same with one level per transferable group (horizon, length class, stops left)
               shared by all routes in the stream
  rolling      the CQR score quantile over the outcomes that arrived in the last `window_days`
               days on the same (route, horizon) stream, once there are at least `min_window`
The interval at miscoverage a is [lo - Q w, hi + Q w], w = max(hi - lo, EPS_WIDTH), where Q is
the 1 - a quantile of the global pool scores of that horizon (aci) or of the window (rolling).
"""

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rail_risk.network_model import calibrate as C

TARGET_MISS = 0.2
ALPHA_BOUNDS = (0.002, 0.998)
WINDOW_DAYS = 14
MIN_WINDOW = 100
METHODS = ("aci_route", "aci_group", "rolling")


def _seconds(ts: pd.Series) -> np.ndarray:
    """Seconds since the epoch whatever the timestamp resolution (the extract uses microseconds)."""
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(float)


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = np.maximum(hi - lo, C.EPS_WIDTH)
    return np.maximum(lo - y, y - hi) / w


@dataclass
class GlobalScores:
    """Sorted pool scores per horizon: the static CQR calibration and its quantile function."""

    by_horizon: dict[int, np.ndarray]

    @staticmethod
    def from_pool(pool: pd.DataFrame) -> "GlobalScores":
        s = cqr_scores(pool["lo"].to_numpy(), pool["hi"].to_numpy(), pool["target_min"].to_numpy())
        h = pool["horizon_min"].to_numpy().astype(int)
        return GlobalScores({int(k): np.sort(s[h == k]) for k in np.unique(h)})

    def q(self, horizon: int, level: float) -> float:
        s = self.by_horizon[int(horizon)]
        return float(np.quantile(s, min(max(level, 0.0), 1.0)))

    def static_q(self, horizon: int) -> float:
        n = len(self.by_horizon[int(horizon)])
        return self.q(horizon, min(1.0, np.ceil((n + 1) * (1 - TARGET_MISS)) / n))


def _run_stream(
    idx: np.ndarray,
    t: np.ndarray,
    arr: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    y: np.ndarray,
    horizon: int,
    method: str,
    gamma: float,
    gs: GlobalScores,
    window_s: float,
    min_window: int,
    out_lo: np.ndarray,
    out_hi: np.ndarray,
    out_alpha: np.ndarray,
) -> None:
    order = np.argsort(t[idx], kind="stable")
    ids = idx[order]
    pending: list[tuple[float, int, int]] = []  # (arrival, tiebreak, row)
    alpha = TARGET_MISS
    window: list[tuple[float, float]] = []  # (arrival, score) of applied outcomes
    q_static = gs.static_q(horizon)
    miss_of: dict[int, float] = {}
    score_of: dict[int, float] = {}
    for k, i in enumerate(ids):
        while pending and pending[0][0] <= t[i]:
            _, _, j = heapq.heappop(pending)
            if method == "rolling":
                window.append((arr[j], score_of[j]))
            else:
                alpha = float(np.clip(alpha + gamma * (TARGET_MISS - miss_of[j]), *ALPHA_BOUNDS))
        w = max(hi[i] - lo[i], C.EPS_WIDTH)
        if method == "rolling":
            cut = t[i] - window_s
            window = [(a, s) for a, s in window if a > cut]
            if len(window) >= min_window:
                sc = np.array([s for _, s in window])
                n = len(sc)
                q = float(np.quantile(sc, min(1.0, np.ceil((n + 1) * (1 - TARGET_MISS)) / n)))
            else:
                q = q_static
            out_alpha[i] = np.nan
        else:
            q = gs.q(horizon, 1.0 - alpha)
            out_alpha[i] = alpha
        out_lo[i], out_hi[i] = lo[i] - q * w, hi[i] + q * w
        miss_of[i] = float(not (out_lo[i] <= y[i] <= out_hi[i]))
        score_of[i] = float(cqr_scores(lo[i : i + 1], hi[i : i + 1], y[i : i + 1])[0])
        heapq.heappush(pending, (arr[i], k, i))


def run(
    rows: pd.DataFrame,
    gs: GlobalScores,
    method: str,
    gamma: float = 0.01,
    window_days: int = WINDOW_DAYS,
    min_window: int = MIN_WINDOW,
) -> pd.DataFrame:
    """rows: route, horizon_min, snap_time, pt_arr, target_min, raw_lo, raw_hi, and for aci_group
    the group columns of calibrate.group_frame. Returns the adapted lo, hi and the miscoverage
    level used for each row (NaN for rolling), aligned to rows."""
    if method not in METHODS:
        raise ValueError(f"unknown adaptive method {method!r}")
    n = len(rows)
    t = _seconds(rows["snap_time"])
    arr = _seconds(rows["pt_arr"]) + rows["target_min"].to_numpy(float) * 60.0
    lo, hi = rows["raw_lo"].to_numpy(float), rows["raw_hi"].to_numpy(float)
    y = rows["target_min"].to_numpy(float)
    h = rows["horizon_min"].to_numpy().astype(int)
    if method == "aci_group":
        g = C.group_frame(rows)
        keys = list(zip(g["h"], g["lc"], g["sb"], strict=True))
    else:
        keys = list(zip(rows["route"], h, strict=True))
    streams: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        streams.setdefault(k, []).append(i)
    out_lo, out_hi, out_alpha = np.empty(n), np.empty(n), np.empty(n)
    for k, ix in streams.items():
        horizon = int(k[0]) if method == "aci_group" else int(k[1])
        _run_stream(
            np.array(ix), t, arr, lo, hi, y, horizon, method, gamma, gs, window_days * 86400.0,
            min_window, out_lo, out_hi, out_alpha,
        )  # fmt: skip
    return pd.DataFrame({"lo": out_lo, "hi": out_hi, "alpha": out_alpha}, index=rows.index)


def block_summary(rows: pd.DataFrame, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Overall coverage, worst route, mean width and coverage in each route's first 7 and last 7
    days of the block, at horizons 15 and 60 and pooled."""
    y = rows["target_min"].to_numpy()
    ok = ((np.asarray(lo) <= y) & (y <= np.asarray(hi))).astype(float)
    wid = np.asarray(hi) - np.asarray(lo)
    h = rows["horizon_min"].to_numpy()
    days = pd.to_datetime(rows["ssd"])
    first = (
        days.groupby(rows["route"].to_numpy())
        .transform(lambda s: s.rank(method="dense"))
        .to_numpy()
    )
    out: dict = {"overall": float(ok.mean()), "mean_width": float(wid.mean())}
    for k in (15, 60):
        m = h == k
        per = pd.Series(ok[m]).groupby(rows["route"].to_numpy()[m]).mean()
        out[f"cov{k}"] = float(ok[m].mean())
        out[f"worst{k}"] = f"{per.idxmin()} {per.min():.3f}"
        out[f"worst{k}_value"] = float(per.min())
    f7 = first <= 7
    per7 = pd.Series(ok[f7]).groupby(rows["route"].to_numpy()[f7]).mean()
    out["first7_overall"] = float(ok[f7].mean())
    out["first7_worst"] = f"{per7.idxmin()} {per7.min():.3f}"
    last = first > first.max() - 7
    out["last7_overall"] = float(ok[last].mean()) if last.any() else float("nan")
    return out
