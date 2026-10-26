"""The freshness guard: how a stale feed changes what the service shows.

One function per target takes ledger columns (everything is known at the snapshot time T) and
returns the final numbers plus the stale flag, its reason and the weight put on the fallback. The
simulation (scripts/stage6_guard_sim.py), the engine and the 2026-10-20 second look all call these,
so what was compared is what is served. Three designs exist; `design` in the guard file picks one:

  widen   keep the model, flag stale, widen the arrival range by the elapsed silence
  binary  fall back to the gap-bucket, the static range and the historical P(late) when stale
  blend   mix model and fallback with a weight that is a logistic function of the silence and is
          zero while the feed is fresh
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def silence(df: pd.DataFrame, window: int) -> np.ndarray:
    """Minutes of silence the service can know at T: the one still running, or the longest one
    inside the last `window` minutes."""
    return np.maximum(df["feed_age_min"].to_numpy(), df[f"feed_gap{window}"].to_numpy())


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class Guard:
    design: str = "binary"
    flag_min: float = 12.0  # silence above this is reported as stale
    flag_window: int = 60
    # widen
    widen_window: int = 60
    widen_k: float = 0.0
    # binary
    bin_min: float = 12.0
    bin_window: int = 60
    # blend: w = gate * sigmoid(a + b1*v60 + b2*v240 + c*(fallback - model)), v = log(silence/flag)
    pair_w: list[float] = field(default_factory=lambda: [-30.0, 0.0, 0.0, 0.0])
    late_w: list[float] = field(default_factory=lambda: [-30.0, 0.0, 0.0, 0.0])
    range_w: list[float] = field(default_factory=lambda: [-30.0, 0.0, 0.0])

    @staticmethod
    def from_json(d: dict) -> "Guard":
        keys = Guard.__dataclass_fields__.keys()
        return Guard(**{k: v for k, v in d.items() if k in keys})

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    # ---- the signal
    def flag(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        s = silence(df, self.flag_window)
        stale = s > self.flag_min
        reason = np.where(stale, "feed silence " + np.round(s).astype(int).astype(str) + " min", "")
        return stale, reason

    def _u(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Log of the silence over the flag threshold, zero when it is at or under it."""
        f = self.flag_min
        return (
            np.log(np.maximum(silence(df, 60), f) / f),
            np.log(np.maximum(silence(df, 240), f) / f),
        )

    def weight(self, df: pd.DataFrame, params: list[float], diff: np.ndarray | None) -> np.ndarray:
        """Zero while the feed is fresh (no silence over flag_min in the last flag_window minutes),
        so a fresh feed serves the model exactly; otherwise a logistic weight on the fallback."""
        u1, u2 = self._u(df)
        z = params[0] + params[1] * u1 + params[2] * u2
        if len(params) > 3 and diff is not None:
            z = z + params[3] * diff
        return _sig(z) * (silence(df, self.flag_window) > self.flag_min)

    # ---- targets
    def pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """p_final, fallback_w, stale and stale_reason for pair rows (needs p_b, p_bar)."""
        stale, reason = self.flag(df)
        pb, pbar = df["p_b"].to_numpy(), df["p_bar"].to_numpy()
        if self.design == "binary":
            s = silence(df, self.bin_window) > self.bin_min
            w = s.astype(float)
        elif self.design == "blend":
            w = self.weight(df, self.pair_w, pbar - pb)
        else:
            w = np.zeros(len(df))
        return pd.DataFrame(
            {"p_final": (1 - w) * pb + w * pbar, "fallback_w": w, "stale": stale,
             "stale_reason": reason},
            index=df.index,
        )  # fmt: skip

    def arrivals(self, df: pd.DataFrame) -> pd.DataFrame:
        """final_lo, final_hi, p_late_final, fallback_w, stale, stale_reason for arrival rows."""
        stale, reason = self.flag(df)
        pt = df["point"].to_numpy()
        lo, hi = df["aci_lo"].to_numpy(), df["aci_hi"].to_numpy()
        slo, shi = df["static_lo"].to_numpy(), df["static_hi"].to_numpy()
        pl, ph = df["p_late"].to_numpy(), df["hist_p"].to_numpy()
        if self.design == "binary":
            w = (silence(df, self.bin_window) > self.bin_min).astype(float)
            wr = w
        elif self.design == "widen":
            s = silence(df, self.widen_window)
            f = np.where(s > self.flag_min, 1.0 + self.widen_k * s / 60.0, 1.0)
            flo, fhi = pt - f * (pt - lo), pt + f * (hi - pt)
            return pd.DataFrame(
                {"final_lo": flo, "final_hi": fhi, "p_late_final": pl,
                 "fallback_w": np.zeros(len(df)), "stale": stale, "stale_reason": reason},
                index=df.index,
            )  # fmt: skip
        elif self.design == "blend":
            w = self.weight(df, self.late_w, ph - pl)
            wr = self.weight(df, self.range_w, None)
        else:
            w = wr = np.zeros(len(df))
        return pd.DataFrame(
            {"final_lo": (1 - wr) * lo + wr * slo, "final_hi": (1 - wr) * hi + wr * shi,
             "p_late_final": (1 - w) * pl + w * ph, "fallback_w": w, "stale": stale,
             "stale_reason": reason},
            index=df.index,
        )  # fmt: skip
