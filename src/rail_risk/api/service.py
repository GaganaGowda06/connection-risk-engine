"""The risk calculation behind the API, with no web framework in it."""

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from rail_risk.api.schemas import DataQuality, DelayRepayOut, RiskResponse
from rail_risk.evaluation import Bands, threshold_probability
from rail_risk.features.snapshots import FEATURE_COLUMNS
from rail_risk.live import FeatureService, LiveStore, resolve_route
from rail_risk.models.bundle import ModelBundle
from rail_risk.repay import DelayRepayRules, RepayError, expected_repay

LONDON = ZoneInfo("Europe/London")
STALE_MINUTES = 10.0
FEATURE_TTL_SECONDS = 60.0
WINDOW_BEFORE_MIN = 20  # keep tracking a service this long after its scheduled arrival
WINDOW_AFTER_MIN = 180


class NotFound(LookupError):
    """The rid is unknown or does not call at a configured route."""


class Warming(Exception):
    """Features for this service are not cached yet; they have been queued to build next."""

    def __init__(self, rid: str, retry_after: int = 2) -> None:
        super().__init__(f"features for {rid} are being built")
        self.rid = rid
        self.retry_after = retry_after


def clock(ts: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(ts).tz_convert(LONDON).strftime("%H:%M")


@dataclass
class Cached:
    features: dict[str, Any]
    minutes: int
    stale: bool
    computed_at: float


@dataclass
class RiskService:
    store: LiveStore
    features: FeatureService
    routes: list[Any]
    bands: Bands
    rules: DelayRepayRules
    bundle: ModelBundle | None
    baseline: Any | None
    model_run_id: str | None
    model_version: str | None
    route_model_trained_on: set[str] = field(default_factory=set)
    now: Any = lambda: datetime.now(UTC)  # noqa: E731 - replaced in tests
    _cache: dict[str, Cached] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _queue: list[str] = field(default_factory=list)  # rids to build first, front of line
    _services: list[dict[str, Any]] = field(default_factory=list)
    _failed: dict[str, float] = field(default_factory=dict)  # rid -> when a build gave nothing
    _memo: dict[Any, Any] | None = None  # prepared tables during one refresh

    # live state

    def age_minutes(self) -> float | None:
        return self.store.data_age_minutes(self.now())

    def is_stale(self) -> bool:
        age = self.age_minutes()
        return age is None or age > STALE_MINUTES

    def active_rids(self) -> list[str]:
        """Services with a scheduled arrival at a route destination in the tracking window."""
        now = pd.Timestamp(self.now())
        dest = {r.destination for r in self.routes}
        tipl = {t for t, crs in self.store.tiplocs.items() if crs in dest}
        ev = self.store.events
        arr = ev[(ev["event_type"] == "arr") & ev["tiploc"].isin(tipl)]
        lo = now - pd.Timedelta(minutes=WINDOW_BEFORE_MIN)
        hi = now + pd.Timedelta(minutes=WINDOW_AFTER_MIN)
        arr = arr[(arr["planned_time"] >= lo) & (arr["planned_time"] <= hi)]
        return sorted(set(arr["rid"]))

    def _prepare(self, rid: str, route: str | None, strip_live: bool):
        key = (rid, route, strip_live)
        if self._memo is not None and key in self._memo:
            hit = self._memo[key]
            if isinstance(hit, NotFound):
                raise hit
            return hit
        try:
            result = self._prepare_uncached(rid, route, strip_live)
        except NotFound as exc:
            if self._memo is not None:
                self._memo[key] = exc
            raise
        if self._memo is not None:
            self._memo[key] = result
        return result

    def _prepare_uncached(self, rid: str, route: str | None, strip_live: bool):
        table = self.store.stop_table(rid)
        if table.empty:
            raise NotFound("no stops known for that service")
        route_id = resolve_route(table, self.routes, prefer=route)
        if route_id is None:
            raise NotFound("service does not call at a configured route")
        dest_crs = next(r.destination for r in self.routes if r.id == route_id)
        dest = table[table["location"] == dest_crs]
        if strip_live:
            table = table.assign(actual_arr=pd.NaT, actual_dep=pd.NaT)
        return table, route_id, dest.iloc[-1]

    def _minutes(self, sched_arr: pd.Timestamp) -> int:
        delta = (sched_arr - pd.Timestamp(self.now())).total_seconds() / 60.0
        return int(round(delta))

    def queue_front(self, rid: str) -> None:
        """Put a rid at the front of the refresher queue (once)."""
        with self._lock:
            if rid not in self._queue:
                self._queue.insert(0, rid)

    def _drain_queue(self) -> list[str]:
        with self._lock:
            queued, self._queue = self._queue, []
        return queued

    def _compute(self, rids: list[str], stale: bool) -> int:
        batch = {}
        for rid in rids:
            try:
                table, route_id, dest = self._prepare(rid, None, strip_live=stale)
            except NotFound:
                continue
            if pd.isna(dest["sched_arr"]):
                continue
            minutes = max(1, self._minutes(dest["sched_arr"]))
            batch[rid] = (table, route_id, minutes)
        out = self.features.batch(batch)
        stamp = time.monotonic()
        with self._lock:
            for rid, row in out.items():
                self._cache[rid] = Cached(row, batch[rid][2], stale, stamp)
                self._failed.pop(rid, None)
            for rid in rids:
                if rid not in out:
                    self._failed[rid] = stamp
        return len(out)

    def refresh_features(self) -> int:
        """Build features for queued rids first, then for every active service.

        A queued rid gets its own small batch ahead of the full one, so a request that missed
        the cache is answered as soon as possible. Returns how many services were built.
        """
        stale = self.is_stale()
        queued = self._drain_queue()
        self._memo = {}
        try:
            built = self._compute(queued, stale) if queued else 0
            rest = [r for r in self.active_rids() if r not in set(queued)]
            built += self._compute(rest, stale)
            self._services = self._service_list()
        finally:
            self._memo = None
        return built

    def _service_list(self) -> list[dict[str, Any]]:
        """Services due at a route destination in the next three hours, for GET /services."""
        now = pd.Timestamp(self.now())
        out = []
        for rid in self.active_rids():
            try:
                table, route_id, dest = self._prepare(rid, None, strip_live=False)
            except NotFound:
                continue
            sched = dest["sched_arr"]
            if pd.isna(sched) or sched < now or sched > now + pd.Timedelta(hours=3):
                continue
            toc = table["toc_code"].dropna()
            first = table.iloc[0]
            dep = first["sched_dep"] if pd.notna(first["sched_dep"]) else first["sched_arr"]
            cancelled, _ = self.store.cancellation(rid)
            out.append(
                {
                    "rid": rid, "route_id": route_id,
                    "toc": None if toc.empty else str(toc.iloc[0]),
                    "origin": str(first["location"]), "scheduled_departure": clock(dep),
                    "scheduled_arrival": clock(sched),
                    "minutes_to_arrival": int(round((sched - now).total_seconds() / 60.0)),
                    "started": bool(pd.notna(first["actual_dep"])), "cancelled": cancelled,
                    "_sort": sched,
                }
            )  # fmt: skip
        out.sort(key=lambda d: d["_sort"])
        return [{k: v for k, v in d.items() if k != "_sort"} for d in out]

    def services(self, route_id: str | None = None) -> list[dict[str, Any]]:
        return [s for s in self._services if route_id is None or s["route_id"] == route_id]

    # scoring

    def _frame(
        self, row: dict[str, Any], route_id: str, horizon: int, minutes: float | None = None
    ) -> pd.DataFrame:
        data = {c: row.get(c) for c in FEATURE_COLUMNS}
        data["route_id"], data["horizon_min"] = route_id, horizon
        if minutes is not None:  # a continuous model sees the real minutes, not the grid value
            data["minutes_to_arrival"] = float(minutes)
        frame = pd.DataFrame([data])
        for col in FEATURE_COLUMNS:
            if col != "toc_code":
                frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
        return frame

    def risk(
        self, rid: str, fare: float | None, ticket: str, route: str | None, request_id: str
    ) -> RiskResponse:
        notes: list[str] = []
        if not self.store.has(rid):
            raise NotFound("unknown service id")
        age, stale = self.age_minutes(), self.is_stale()
        if stale:
            notes.append(
                "Darwin data is older than "
                f"{STALE_MINUTES:.0f} minutes, so the service is treated as not started"
            )
        cancelled, reason = self.store.cancellation(rid)
        table, route_id, dest = self._prepare(rid, route, strip_live=stale)
        quality = dict(
            live_data_age_minutes=None if age is None else round(age, 1),
            stale_live_data=stale,
            model_fallback=self.bundle is None,
            horizon_snapped_from_minutes=None,
            horizon_used_minutes=None,
            route_in_training=route_id in self.route_model_trained_on,
            schedule_known=rid in self.store.schedules,
            features_age_seconds=None,
        )
        toc = None if table["toc_code"].isna().all() else str(table["toc_code"].dropna().iloc[0])
        sched = dest["sched_arr"]
        base = dict(
            request_id=request_id, rid=rid, route_id=route_id, toc=toc,
            scheduled_arrival=None if pd.isna(sched) else clock(sched),
            model_run_id=None if self.bundle is None else self.model_run_id,
            model_version=None if self.bundle is None else self.model_version,
        )  # fmt: skip
        if cancelled:
            return RiskResponse(
                **base, status="cancelled", arrival_range_80=None, median_arrival=None,
                median_delay_minutes=None, p_delay_at_least_15=None, p_delay_at_least_30=None,
                p_delay_at_least_60=None, minutes_to_arrival_used=None,
                cancellation_reason=reason, delay_repay=None,
                data_quality=DataQuality(**quality, notes=[*notes, "service is cancelled"]),
            )  # fmt: skip
        if pd.notna(dest["actual_arr"]) and not stale:
            delay = (dest["actual_arr"] - sched).total_seconds() / 60.0
            notes.append(f"the service has arrived, {delay:+.0f} minutes against schedule")
            return RiskResponse(
                **base, status="arrived", arrival_range_80=None,
                median_arrival=clock(dest["actual_arr"]),
                median_delay_minutes=delay, p_delay_at_least_15=float(delay >= 15),
                p_delay_at_least_30=float(delay >= 30), p_delay_at_least_60=float(delay >= 60),
                minutes_to_arrival_used=None, delay_repay=self._repay(
                    toc, fare, ticket, self._certain_probs(delay)
                ),
                data_quality=DataQuality(**quality, notes=notes),
            )  # fmt: skip
        if pd.isna(sched):
            raise NotFound("service has no scheduled arrival at its destination")

        minutes = max(1, self._minutes(sched))
        features, features_age = self._features_for(rid, table, route_id, minutes, stale)
        quality["features_age_seconds"] = None if features_age is None else round(features_age, 1)
        if not features:
            raise NotFound("features could not be built for that service")
        if not quality["route_in_training"] and self.route_model_trained_on:
            notes.append(
                f"the model was not trained on route {route_id}, so its accuracy there is "
                "not validated"
            )
        if toc is None:
            notes.append("no schedule message seen, so the operator is unknown")
        point, lo, hi, probs, used = self._predict(features, route_id, minutes, quality, notes)
        started = pd.notna(features.get("last_stop_lateness_min"))
        return RiskResponse(
            **base,
            status="running" if started else "not_started",
            arrival_range_80=f"{clock(sched + pd.Timedelta(minutes=lo))} to "
            f"{clock(sched + pd.Timedelta(minutes=hi))}",
            median_arrival=clock(sched + pd.Timedelta(minutes=point)),
            median_delay_minutes=round(float(point), 1),
            p_delay_at_least_15=round(float(threshold_probability(probs, self.bands, 15)[0]), 4),
            p_delay_at_least_30=round(float(threshold_probability(probs, self.bands, 30)[0]), 4),
            p_delay_at_least_60=round(float(threshold_probability(probs, self.bands, 60)[0]), 4),
            minutes_to_arrival_used=float(minutes),
            delay_repay=self._repay(toc, fare, ticket, probs[0]),
            data_quality=DataQuality(**quality, notes=notes),
        )

    def _features_for(self, rid, table, route_id, minutes, stale):
        with self._lock:
            hit = self._cache.get(rid)
        if (
            hit is not None
            and hit.stale == stale
            and abs(hit.minutes - minutes) <= 1
            and time.monotonic() - hit.computed_at <= FEATURE_TTL_SECONDS
        ):
            return hit.features, time.monotonic() - hit.computed_at
        with self._lock:
            failed_at = self._failed.get(rid)
        if failed_at is not None and time.monotonic() - failed_at < FEATURE_TTL_SECONDS * 2:
            raise NotFound("features could not be built for that service")
        # never build on the request path: a Spark job would blow the budget
        self.queue_front(rid)
        raise Warming(rid)

    def _predict(self, features, route_id, minutes, quality, notes):
        if self.bundle is not None:
            horizon, snapped = self.bundle.model_horizon(minutes)
            quality["horizon_used_minutes"] = int(horizon)
            if snapped:
                quality["horizon_snapped_from_minutes"] = float(minutes)
                notes.append(f"the model was given its {horizon} minute horizon, not {minutes}")
            real = minutes if self.bundle.continuous_horizon else None
            frame = self._frame(features, route_id, horizon, real)
            point, lo, hi = self.bundle.quantile.predict(frame)
            probs = self.bundle.bands.predict(frame)
            return float(point[0]), float(lo[0]), float(hi[0]), probs, horizon
        frame = self._frame(features, route_id, minutes)
        notes.append("model unavailable: historical baseline served")
        if self.baseline is not None:
            pred = self.baseline.predict(frame.assign(dest_hour=features.get("dest_hour")))
            return float(pred.point[0]), float(pred.lo[0]), float(pred.hi[0]), pred.probs, minutes
        zeros = np.zeros((1, self.bands.k))
        zeros[0, 0] = 1.0
        notes.append("no baseline available: timetable (on time) served")
        return 0.0, 0.0, 0.0, zeros, minutes

    def _certain_probs(self, delay: float) -> np.ndarray:
        idx = int(np.digitize([delay], list(self.bands.lower_edges)[1:])[0])
        probs = np.zeros(self.bands.k)
        probs[idx] = 1.0
        return probs

    def _repay(self, toc, fare, ticket, probs) -> DelayRepayOut:
        if toc is None:
            return DelayRepayOut(
                toc=None, toc_status="unknown", amount_gbp=None,
                reason="the operator is not known, so no rules apply",
            )  # fmt: skip
        op = self.rules.operators.get(toc)
        if op is None:
            return DelayRepayOut(
                toc=toc, toc_status="unknown", amount_gbp=None,
                reason=f"no Delay Repay rules for {toc}",
            )  # fmt: skip
        if op.status != "verified":
            return DelayRepayOut(
                toc=toc, toc_status="unverified", amount_gbp=None,
                reason=f"{op.name} Delay Repay tiers are unverified", caveat=op.caveat,
            )  # fmt: skip
        if fare is None:
            return DelayRepayOut(
                toc=toc, toc_status="verified", amount_gbp=None,
                reason="no fare supplied", caveat=op.caveat,
            )  # fmt: skip
        try:
            result = expected_repay(
                np.asarray(probs, dtype=float).ravel(),
                self.bands.lower_edges,
                toc,
                ticket,
                fare,
                self.rules,
            )
        except RepayError as exc:
            return DelayRepayOut(
                toc=toc, toc_status="verified", amount_gbp=None, reason=str(exc),
                caveat=op.caveat,
            )  # fmt: skip
        return DelayRepayOut(
            toc=toc, toc_status="verified", amount_gbp=round(result.expected_payout, 2),
            reason=None, caveat=op.caveat,
        )  # fmt: skip
