"""Online recalibration of P(miss), one bias per hub, starting from the global model.

A scored row at time T gets p' = sigmoid(logit(p) + b), where b is the hub's current bias, and b
starts at 0 (the global calibrated model) for every hub at the start of a block. An outcome (a
labelled row) changes b only after its outbound has actually departed, and only for rows scored at
a later time: time moves in blocks of `block_s` seconds, and a block uses the outcomes whose
departure time is strictly before the block starts. The update is a log-loss gradient step,
b <- b - eta * sum(p' - y) over the outcomes that just became known, with p' recomputed from the
current b. Cancelled outbounds have no label and never update anything.
"""

import numpy as np
import pandas as pd

BLOCK_S = 60.0
ETAS = (0.0, 1e-4, 3e-4, 1e-3)
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def seconds(ts: pd.Series) -> np.ndarray:
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(float)


def run(
    hub: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    y: np.ndarray,
    resolve: np.ndarray,
    eta: float,
    block_s: float = BLOCK_S,
) -> tuple[np.ndarray, np.ndarray]:
    """t, resolve: epoch seconds. y is NaN for rows with no label (never an update).
    Returns the adapted probability and, per row, the latest departure time among the outcomes
    that had been applied when it was scored (NaN if none), which the leakage test checks."""
    out = p.copy()
    last_used = np.full(len(p), np.nan)
    if eta == 0.0:
        return out, last_used
    for h in np.unique(hub):
        idx = np.flatnonzero(hub == h)
        order = idx[np.argsort(t[idx], kind="stable")]
        ev = idx[np.isfinite(y[idx]) & np.isfinite(resolve[idx])]
        ev = ev[np.argsort(resolve[ev], kind="stable")]
        ev_r = resolve[ev]
        blocks = np.floor(t[order] / block_s).astype(np.int64)
        starts = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1]])
        ends = np.r_[starts[1:], len(order)]
        b, ptr, latest = 0.0, 0, np.nan
        for s, e in zip(starts, ends, strict=True):
            block_start = blocks[s] * block_s
            nxt = np.searchsorted(ev_r, block_start, side="left")  # departures strictly before
            if nxt > ptr:
                e_ix = ev[ptr:nxt]
                pe = _sig(_logit(p[e_ix]) + b)
                b -= eta * float(np.sum(pe - y[e_ix]))
                latest = ev_r[nxt - 1]
                ptr = nxt
            rows = order[s:e]
            out[rows] = _sig(_logit(p[rows]) + b)
            last_used[rows] = latest
    return out, last_used
