"""Connection simulator: per-leg predictions to a journey-level miss probability and payout.

An inbound leg arrives at the hub with a predictive delay distribution (the calibrated quantiles
of the Stage 4 model, joined by monotone interpolation). The planned outbound train and the next
few alternatives each get an empirical delay draw from history. A draw misses when the inbound
arrival plus the minimum connection time is after the outbound's actual departure, and then the
first catchable alternative carries the passenger. Times are minutes from local midnight of the
service date.

Legs are treated as independent: the inbound delay and the outbound delays are drawn separately,
so shared causes such as a signalling failure or a strike day are not modelled. In the
history the correlation is weak (0.11) and, for tight changes, independence overstated misses.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from rail_risk.repay import DelayRepayRules, ExpectedRepay, expected_repay

LONDON = ZoneInfo("Europe/London")
DEFAULT_DRAWS = 2000
DEFAULT_SEED = 20261015
MAX_FLOOR_LEVEL = 0.995  # a floor above this level of the distribution is held to it


# stations


class Station(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    minutes: int
    status: str  # "verified" or "assumed"
    source: str
    comment: str | None = None


def load_stations(path: Path | str = "config/stations.yaml") -> dict[str, Station]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {k: Station.model_validate(v) for k, v in raw["stations"].items()}


# outbound delay model


@dataclass(frozen=True)
class Train:
    """A planned outbound train: scheduled departure from the hub, scheduled final arrival."""

    sched_dep: float
    sched_arr: float
    toc: str | None = None

    @property
    def hour(self) -> int:
        return int(self.sched_dep // 60) % 24


class OutboundDelays:
    """Joint empirical draws of (departure delay at the hub, arrival delay at the final stop).

    Pairs are kept together so a train that leaves late tends to arrive late, as it did in the
    history. They are grouped by scheduled departure hour. An hour with fewer than `min_rows`
    pairs falls back to every hour pooled.
    """

    def __init__(self, by_hour: dict[int, np.ndarray], min_rows: int = 30) -> None:
        pooled = [a for a in by_hour.values() if len(a)]
        if not pooled:
            raise ValueError("no outbound history to draw from")
        self.pooled = np.vstack(pooled)
        self.by_hour = {h: a for h, a in by_hour.items() if len(a) >= min_rows}
        self.min_rows = min_rows

    @classmethod
    def fit(cls, services: pd.DataFrame, min_rows: int = 30) -> "OutboundDelays":
        """`services` has columns hour, dep_delay and arr_delay (minutes), training dates only."""
        df = services.dropna(subset=["hour", "dep_delay", "arr_delay"])
        by_hour = {
            int(h): g[["dep_delay", "arr_delay"]].to_numpy(dtype=float)
            for h, g in df.groupby("hour")
        }
        return cls(by_hour, min_rows)

    def sample(self, hour: int, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        pairs = self.by_hour.get(int(hour), self.pooled)
        pick = pairs[rng.integers(0, len(pairs), n)]
        return pick[:, 0], pick[:, 1]


class FixedDelays(OutboundDelays):
    """Every outbound train has the same delays. For tests and what-if cases."""

    def __init__(self, dep_delay: float = 0.0, arr_delay: float = 0.0) -> None:
        self.pooled = np.array([[dep_delay, arr_delay]])
        self.by_hour = {}
        self.min_rows = 1


# inbound distribution


def inbound_delay_draws(
    levels: np.ndarray, quantiles: np.ndarray, u: np.ndarray, tail: np.ndarray | None = None
) -> np.ndarray:
    """Inverse CDF by linear interpolation across the quantiles.

    Above the top level the draw is the top quantile plus a point on the empirical quantile
    function of `tail`, the amounts by which training outcomes exceeded the top quantile. With no
    tail it falls back to extending the last segment linearly, which understates long delays.
    The lower end is extended linearly by one segment.
    """
    levels = np.asarray(levels, dtype=float)
    q = np.maximum.accumulate(np.asarray(quantiles, dtype=float))  # monotone
    x = np.interp(u, levels, q)
    lo, hi = u < levels[0], u > levels[-1]
    if lo.any():
        slope = (q[1] - q[0]) / (levels[1] - levels[0])
        x[lo] = q[0] - (levels[0] - u[lo]) * slope
    if hi.any():
        if tail is not None and len(tail) > 0:
            position = (u[hi] - levels[-1]) / (1.0 - levels[-1])
            x[hi] = q[-1] + np.quantile(np.sort(np.asarray(tail, dtype=float)), position)
        else:
            slope = (q[-1] - q[-2]) / (levels[-1] - levels[-2])
            x[hi] = q[-1] + (u[hi] - levels[-1]) * slope
    return x


class TailModel:
    """Empirical excess over the top predicted quantile, per observed-lateness group.

    Fitted from training rows only: for each row whose outcome exceeded its calibrated top
    quantile, the excess is outcome minus quantile. A group with fewer than `min_rows` excesses
    uses the pooled excesses of all groups.
    """

    def __init__(self, by_group: dict[int, np.ndarray], min_rows: int = 30) -> None:
        arrays = [np.asarray(a, dtype=float) for a in by_group.values() if len(a)]
        self.pooled = np.sort(np.concatenate(arrays)) if arrays else np.empty(0)
        self.by_group = {
            int(g): np.sort(np.asarray(a, dtype=float))
            for g, a in by_group.items()
            if len(a) >= min_rows
        }

    def for_group(self, group: int) -> np.ndarray:
        return self.by_group.get(int(group), self.pooled)

    @classmethod
    def fit(
        cls, outcome: np.ndarray, top_quantile: np.ndarray, groups: np.ndarray, min_rows: int = 30
    ) -> "TailModel":
        excess = np.asarray(outcome, dtype=float) - np.asarray(top_quantile, dtype=float)
        by_group = {int(g): excess[(groups == g) & (excess > 0)] for g in np.unique(groups)}
        return cls(by_group, min_rows)


# simulation


@dataclass
class ConnectionResult:
    p_miss: float
    p_stranded: float
    stranded: bool  # no draw reached the final stop: there is no arrival to describe
    n_draws: int
    delay_quantiles: dict[float, float] | None  # final arrival delay vs the planned arrival
    band_probs: np.ndarray | None  # over delay bands; stranded draws count in the top band
    delays: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    planned_arrival: float = 0.0
    min_connection_time: float = 0.0

    def arrival_range(self, lo: float = 0.1, hi: float = 0.9) -> tuple[float, float] | None:
        """(early, late) final arrival as minutes from midnight, or None when stranded."""
        if self.delay_quantiles is None:
            return None
        return (
            self.planned_arrival + self.delay_quantiles[lo],
            self.planned_arrival + self.delay_quantiles[hi],
        )


def simulate(
    levels: np.ndarray,
    quantiles: np.ndarray,
    inbound_sched_arr: float,
    planned: Train,
    alternatives: Sequence[Train],
    min_connection_time: float,
    outbound: OutboundDelays,
    band_edges: Sequence[int] = (0, 15, 30, 60, 120),
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    tail: np.ndarray | None = None,
    inbound_floor: float | None = None,
) -> ConnectionResult:
    """Monte Carlo over the inbound arrival and the outbound trains.

    The random numbers depend only on the seed and the number of trains, so changing the
    minimum connection time or shifting the inbound quantiles reuses the same draws and the
    miss probability moves monotonically.

    `inbound_floor` conditions the inbound delay on being at least that many minutes, for a
    forecast made after the scheduled arrival with the train still not in: the uniform draws are
    squeezed into the part of the distribution above the floor.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(draws)
    if inbound_floor is not None:
        q_sorted = np.maximum.accumulate(np.asarray(quantiles, dtype=float))
        below = float(np.interp(inbound_floor, q_sorted, np.asarray(levels, dtype=float)))
        below = min(max(below, 0.0), MAX_FLOOR_LEVEL)
        u = below + (1.0 - below) * u
    # Times are whole minutes in the data, so a delay is too. A continuous draw would split every
    # tie between a late inbound and a late outbound as half a miss, where the data counts a tie
    # as made (ready at or before departure). Rounding puts the draws on the data's grid.
    arrival = inbound_sched_arr + np.rint(inbound_delay_draws(levels, quantiles, u, tail))
    trains = [planned, *alternatives]
    dep_actual, arr_actual = [], []
    for train in trains:
        d, a = outbound.sample(train.hour, rng, draws)
        dep_actual.append(train.sched_dep + d)
        arr_actual.append(train.sched_arr + a)
    ready = arrival + min_connection_time
    caught = [ready <= dep for dep in dep_actual]

    missed = ~caught[0]
    final = arr_actual[0].copy()
    completed = caught[0].copy()
    for i in range(1, len(trains)):
        take = ~completed & caught[i]
        final[take] = arr_actual[i][take]
        completed |= take
    delays = (final - planned.sched_arr)[completed]
    # a stranded draw has no arrival; for the payout it counts in the top tier (see the note)
    all_delays = np.where(completed, final - planned.sched_arr, np.inf)
    p_stranded = float(1.0 - completed.mean())
    stranded = not completed.any()
    quantile_levels = (0.1, 0.5, 0.9)
    result = ConnectionResult(
        p_miss=float(missed.mean()),
        p_stranded=p_stranded,
        stranded=stranded,
        n_draws=draws,
        delay_quantiles=None
        if stranded
        else {q: float(np.quantile(delays, q)) for q in quantile_levels},
        band_probs=band_probabilities(all_delays, band_edges),
        delays=delays,
        planned_arrival=planned.sched_arr,
        min_connection_time=float(min_connection_time),
    )
    return result


def band_probabilities(delays: np.ndarray, lower_edges: Sequence[int]) -> np.ndarray:
    """Share of delays in each band. Early arrivals fall in the first band."""
    idx = np.digitize(np.asarray(delays, dtype=float), list(lower_edges)[1:])
    return np.bincount(idx, minlength=len(lower_edges)) / len(delays)


def journey_repay(
    result: ConnectionResult,
    lower_edges: Sequence[int],
    toc: str,
    ticket: str,
    fare: float,
    rules: DelayRepayRules,
) -> ExpectedRepay:
    """Expected Delay Repay for the journey over all draws.

    Stranded draws (no catchable train that day) are priced in the top tier. That is an
    assumption: it treats a stranded passenger as claiming the longest delay.
    """
    return expected_repay(result.band_probs, tuple(lower_edges), toc, ticket, fare, rules)


# feasibility


class InfeasibleConnection(ValueError):
    """The planned connection cannot be made on the timetable at the minimum connection time."""

    def __init__(self, message: str, first_feasible: Train | None) -> None:
        super().__init__(message)
        self.first_feasible = first_feasible


def first_feasible(
    inbound_sched_arr: float, trains: Sequence[Train], min_connection_time: float
) -> Train | None:
    """The first outbound whose scheduled departure leaves at least the minimum connection time."""
    for train in sorted(trains, key=lambda t: t.sched_dep):
        if train.sched_dep - inbound_sched_arr >= min_connection_time:
            return train
    return None


def choose_outbound(
    inbound_sched_arr: float,
    trains: Sequence[Train],
    min_connection_time: float,
    planned_dep: float | None = None,
) -> Train:
    """The outbound to use. None given: the first feasible. One given: it must be feasible.

    An infeasible planned departure is refused with a message naming the first feasible train.
    """
    first = first_feasible(inbound_sched_arr, trains, min_connection_time)
    if planned_dep is None:
        if first is None:
            raise InfeasibleConnection(
                "no outbound departs after the minimum connection time", None
            )
        return first
    match = [t for t in trains if abs(t.sched_dep - planned_dep) < 0.5]
    if not match:
        raise InfeasibleConnection("no planned outbound departs at that time", first)
    chosen = match[0]
    gap = chosen.sched_dep - inbound_sched_arr
    if gap < min_connection_time:
        named = (
            f"the first feasible outbound departs {minutes_to_hhmm(first.sched_dep)}"
            if first is not None
            else "no later outbound is listed"
        )
        raise InfeasibleConnection(
            f"infeasible on paper: the {minutes_to_hhmm(chosen.sched_dep)} leaves {gap:.0f} min "
            f"after the inbound is due, less than the {min_connection_time:.0f} min minimum "
            f"connection time; {named}",
            first,
        )
    return chosen


# data


def to_local_minutes(ts: pd.Series, day: date) -> pd.Series:
    """Minutes from local midnight of `day` for UTC timestamps."""
    local = pd.to_datetime(ts, utc=True).dt.tz_convert(LONDON)
    midnight = pd.Timestamp(day).tz_localize(LONDON)
    return (local - midnight).dt.total_seconds() / 60.0


def service_table(
    stops: pd.DataFrame, hub: str, destination: str, day_of: dict[str, date] | None = None
) -> pd.DataFrame:
    """One row per service that calls at both stations: hub departure and destination arrival.

    Columns: rid, date, toc, sched_dep, sched_arr (local minutes from midnight of the service
    date), dep_delay, arr_delay (minutes) and hour (scheduled departure hour).
    """
    at_hub = stops[stops["location"] == hub].drop_duplicates("rid")
    at_dest = stops[stops["location"] == destination].drop_duplicates("rid")
    merged = at_hub[["rid", "toc_code", "sched_dep", "actual_dep", "date_of_service"]].merge(
        at_dest[["rid", "sched_arr", "actual_arr"]], on="rid"
    )
    merged["date"] = pd.to_datetime(merged["date_of_service"]).dt.date
    out = []
    for day, g in merged.groupby("date"):
        g = g.copy()
        g["dep_delay"] = (
            pd.to_datetime(g["actual_dep"], utc=True) - pd.to_datetime(g["sched_dep"], utc=True)
        ).dt.total_seconds() / 60.0
        g["arr_delay"] = (
            pd.to_datetime(g["actual_arr"], utc=True) - pd.to_datetime(g["sched_arr"], utc=True)
        ).dt.total_seconds() / 60.0
        g["sched_dep_min"] = to_local_minutes(g["sched_dep"], day)
        g["sched_arr_min"] = to_local_minutes(g["sched_arr"], day)
        out.append(g)
    if not out:
        return pd.DataFrame(
            columns=[
                "rid",
                "date",
                "toc",
                "sched_dep",
                "sched_arr",
                "dep_delay",
                "arr_delay",
                "hour",
            ]
        )
    df = pd.concat(out, ignore_index=True)
    df = df.rename(columns={"toc_code": "toc", "sched_dep_min": "sched_dep_m"})
    df["sched_dep"], df["sched_arr"] = df["sched_dep_m"], df["sched_arr_min"]
    df["hour"] = (df["sched_dep"] // 60).astype("Int64") % 24
    return df[["rid", "date", "toc", "sched_dep", "sched_arr", "dep_delay", "arr_delay", "hour"]]


def hhmm_to_minutes(text: str) -> float:
    hh, mm = text.split(":")
    if not (0 <= int(hh) < 24 and 0 <= int(mm) < 60):
        raise ValueError(f"not a clock time: {text!r}")
    return int(hh) * 60.0 + int(mm)


def minutes_to_hhmm(minutes: float) -> str:
    total = int(round(minutes))
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"
