"""Live evaluation of the frozen champion on captured Darwin data, as a point-in-time replay.

The API did not log its predictions, so each prediction is rebuilt after the fact. For a service
and a prediction time T the live state is rebuilt from the Darwin messages whose message time is at
or before T and nothing else: events, schedule versions (so a cancellation that arrived after T is
unknown at T) and the age of the newest message. The stop table, the features and the prediction
then come from the same functions the API calls (`LiveStore.stop_table`, `FeatureService.batch`,
`RiskService._predict`, `RiskService.connection`). Nothing here fits or changes a model.

A capture gap is a stretch of message time longer than five minutes with no message in it. A
service whose prediction-to-arrival window overlaps a gap is excluded, because what the live
service would have known there was lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from rail_risk.live import DEDUP_KEY, EVENT_COLUMNS, LiveStore, read_parts, resolve_route

LONDON = ZoneInfo("Europe/London")
MAX_GAP = pd.Timedelta(minutes=5)
NS_MIN = 60 * 10**9

# capture gaps


def find_gaps(
    stamps: np.ndarray, start: pd.Timestamp, end: pd.Timestamp, max_gap: pd.Timedelta = MAX_GAP
) -> pd.DataFrame:
    """Stretches inside [start, end] with no message for longer than `max_gap`.

    `stamps` is every captured message time as int64 nanoseconds since the epoch, sorted. A gap
    runs from one message to the next and is clipped to the window. A window with no message at
    all is one gap. Columns: gap_start, gap_end (UTC) and minutes.
    """
    s, e = start.value, end.value
    ordered = np.sort(np.asarray(stamps, dtype="int64"))
    ordered = np.unique(ordered)
    before = ordered[ordered <= s]
    after = ordered[ordered >= e]
    inside = ordered[(ordered > s) & (ordered < e)]
    # the neighbours just outside the window decide whether the edges are gaps
    edges = np.concatenate(
        [before[-1:] if len(before) else [s], inside, after[:1] if len(after) else [e]]
    )
    diffs = np.diff(edges)
    bad = np.flatnonzero(diffs > max_gap.value)
    rows = [(max(int(edges[i]), s), min(int(edges[i + 1]), e)) for i in bad]
    frame = pd.DataFrame(rows, columns=["gap_start", "gap_end"], dtype="int64")
    out = pd.DataFrame(
        {
            "gap_start": pd.to_datetime(frame["gap_start"], utc=True),
            "gap_end": pd.to_datetime(frame["gap_end"], utc=True),
        }
    )
    out["minutes"] = (frame["gap_end"] - frame["gap_start"]) / NS_MIN
    return out


def gap_hours_by_day(
    gaps: pd.DataFrame, days: list[date], hours: tuple[int, int] = (0, 24)
) -> dict[date, float]:
    """Hours of capture missing on each local calendar day, splitting a gap at midnight.

    `hours` limits the count to local clock hours [start, end) of each day, for example the hours
    the services run.
    """
    out = dict.fromkeys(days, 0.0)
    for g in gaps.itertuples():
        a, b = g.gap_start.tz_convert(LONDON), g.gap_end.tz_convert(LONDON)
        day = a.normalize()
        while day < b:
            nxt = (day + pd.Timedelta(days=1)).normalize()
            lo = max(a, day + pd.Timedelta(hours=hours[0]))
            hi = min(b, day + pd.Timedelta(hours=hours[1]), nxt)
            if hi > lo and day.date() in out:
                out[day.date()] += (hi - lo).total_seconds() / 3600.0
            day = nxt
    return out


class GapIndex:
    """Fast overlap test of a window against a set of gaps."""

    def __init__(self, gaps: pd.DataFrame) -> None:
        start, end = _ns(gaps["gap_start"]), _ns(gaps["gap_end"])
        order = np.argsort(start)
        self.start, self.end = start[order], end[order]

    def overlaps(self, t0: pd.Timestamp, t1: pd.Timestamp) -> bool:
        """True when any gap shares time with the window from t0 to t1 (touching does not count)."""
        a, b = t0.value, t1.value
        return bool(np.any((self.start < b) & (self.end > a)))


# point-in-time state


def _ns(series: pd.Series) -> np.ndarray:
    """Nanoseconds since the epoch, whatever time unit pandas holds the column in."""
    return pd.to_datetime(series, utc=True).dt.as_unit("ns").astype("int64").to_numpy()


@dataclass
class ReplayData:
    """Every captured Darwin message of the watched services, indexed for as-of reads."""

    events: dict[str, tuple[pd.DataFrame, np.ndarray]]  # rid -> (events, message time ns)
    schedules: dict[str, tuple[list[dict[str, Any]], np.ndarray]]
    stamps: np.ndarray  # every message time seen, ns, sorted and unique
    tiplocs: dict[str, str]
    days: list[date] = field(default_factory=list)

    def store_at(self, rids: list[str], t: pd.Timestamp | None) -> LiveStore:
        """A LiveStore holding only what was known at `t`. `t=None` is the whole capture.

        Events and schedule versions with a message time after `t` are not in the store, and the
        newest message time is the newest captured at or before `t`.
        """
        cut = None if t is None else pd.Timestamp(t).value
        store = LiveStore(Path("."), Path("."), self.tiplocs)
        parts: dict[str, pd.DataFrame] = {}
        for rid in rids:
            if rid not in self.events:
                continue
            frame, ns = self.events[rid]
            k = len(frame) if cut is None else int(np.searchsorted(ns, cut, side="right"))
            if k:
                parts[rid] = frame.iloc[:k]
            versions, sns = self.schedules.get(rid, ([], np.empty(0, dtype="int64")))
            j = len(versions) if cut is None else int(np.searchsorted(sns, cut, side="right"))
            if j:
                store.schedules[rid] = versions[j - 1]
        store.events = (
            pd.concat(parts.values(), ignore_index=True)
            if parts
            else pd.DataFrame(columns=EVENT_COLUMNS)
        )
        store._by_rid = parts
        if cut is None:
            newest = int(self.stamps[-1]) if len(self.stamps) else None
        else:
            i = int(np.searchsorted(self.stamps, cut, side="right"))
            newest = int(self.stamps[i - 1]) if i else None
        store.newest_message_ts = (
            None if newest is None else pd.Timestamp(newest, tz="UTC").to_pydatetime()
        )
        return store


def load_replay_data(interim: Path, days: list[date], tiplocs: dict[str, str]) -> ReplayData:
    """Read the parsed Darwin tables for the service dates in `days`.

    Events are kept for every service that calls at a watched TIPLOC, deduplicated on the Kafka
    key, and ordered by message time. Message times of every captured message, watched or not,
    are kept for gap detection and the age of the live data.
    """
    root = interim / "darwin"
    watch = list(tiplocs)
    stamps = []
    for table in ("darwin_events", "darwin_schedules"):
        for part in sorted((root / table).glob("date=*")):
            df = read_parts(part, ["message_ts"])
            if len(df):
                stamps.append(np.unique(_ns(df["message_ts"])))
    all_stamps = np.unique(np.concatenate(stamps)) if stamps else np.empty(0, dtype="int64")

    keep: list[pd.DataFrame] = []
    sched_rows: list[pd.DataFrame] = []
    for day in days:
        events_dir = root / "darwin_events" / f"date={day.isoformat()}"
        if not events_dir.exists():
            continue
        df = read_parts(events_dir, [c for c in EVENT_COLUMNS if c != "ssd"])
        if df.empty:
            continue
        rids = set(df.loc[df["tiploc"].isin(watch), "rid"])
        df = df[df["rid"].isin(rids)].assign(ssd=pd.Timestamp(day))
        keep.append(df[EVENT_COLUMNS])
        sched_dir = root / "darwin_schedules" / f"date={day.isoformat()}"
        if sched_dir.exists():
            sched = read_parts(sched_dir)
            sched_rows.append(sched[sched["rid"].isin(rids)])
    events = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=EVENT_COLUMNS)
    events = events.drop_duplicates(DEDUP_KEY, keep="last")
    events = events.sort_values(["message_ts", "offset"], kind="stable")
    by_rid = {
        str(rid): (g.reset_index(drop=True), _ns(g["message_ts"]))
        for rid, g in events.groupby("rid", sort=False)
    }
    schedules: dict[str, tuple[list[dict[str, Any]], np.ndarray]] = {}
    if sched_rows:
        sched_all = pd.concat(sched_rows, ignore_index=True)
        sched_all = sched_all.sort_values(["message_ts", "offset"], kind="stable")
        for rid, g in sched_all.groupby("rid", sort=False):
            schedules[str(rid)] = (g.to_dict("records"), _ns(g["message_ts"]))
    return ReplayData(by_rid, schedules, all_stamps, tiplocs, list(days))


# the service, the services in the window and the features


def _crs_tiplocs(tiplocs: dict[str, str], crs: str) -> set[str]:
    return {t for t, c in tiplocs.items() if c == crs}


def _last_message(ev: pd.DataFrame, tiplocs: set[str], kind: str, event: str) -> pd.Timestamp:
    """Message time of the newest `kind` time of an arrival or departure at the TIPLOCs."""
    hit = ev[ev["tiploc"].isin(tiplocs) & (ev["event_type"] == event) & (ev["time_kind"] == kind)]
    return hit["message_ts"].max() if len(hit) else pd.NaT


def build_universe(
    data: ReplayData, routes: list[Any], route_ids: tuple[str, ...], days: list[date]
) -> pd.DataFrame:
    """One row per (service, route) in the window, from the whole capture.

    This is the outcome side only: scheduled and actual times and whether the service ended up
    cancelled. It decides which services exist and what happened. It is never used to build a
    feature. Columns: rid, route_id, date, toc, sched_arr, actual_arr, actual_msg_ts (message
    time that carried the actual arrival), cancelled, and the hub (BHM) departure and arrival
    times for the connection pairs.
    """
    dest_crs = {r.id: r.destination for r in routes}
    dest_tpl = {rid: _crs_tiplocs(data.tiplocs, crs) for rid, crs in dest_crs.items()}
    hub_tpl = _crs_tiplocs(data.tiplocs, "BHM")
    window = set(days)
    rows = []
    for rid in data.events:
        store = data.store_at([rid], None)
        table = store.stop_table(rid)
        if table.empty:
            continue
        ssd = pd.Timestamp(table["date_of_service"].iloc[0]).date()
        if ssd not in window:
            continue
        cancelled, _ = store.cancellation(rid)
        ev = store.events
        toc = table["toc_code"].dropna()
        for route_id in route_ids:
            if resolve_route(table, routes, prefer=route_id) is None:
                continue
            dest = table[table["location"] == dest_crs[route_id]].iloc[-1]
            if pd.isna(dest["sched_arr"]):
                continue
            hub = table[table["location"] == "BHM"]
            hub_row = hub.iloc[-1] if len(hub) else None
            rows.append(
                {
                    "rid": rid, "route_id": route_id, "date": ssd,
                    "toc": None if toc.empty else str(toc.iloc[0]),
                    "sched_arr": dest["sched_arr"], "actual_arr": dest["actual_arr"],
                    "actual_msg_ts": _last_message(ev, dest_tpl[route_id], "actual", "arr"),
                    "cancelled": bool(cancelled),
                    "hub_sched_dep": pd.NaT if hub_row is None else hub_row["sched_dep"],
                    "hub_actual_dep": pd.NaT if hub_row is None else hub_row["actual_dep"],
                    "hub_dep_msg_ts": _last_message(ev, hub_tpl, "actual", "dep"),
                }
            )  # fmt: skip
    out = pd.DataFrame(rows)
    for col in ("sched_arr", "actual_arr", "actual_msg_ts", "hub_sched_dep", "hub_actual_dep",
                "hub_dep_msg_ts"):  # fmt: skip
        out[col] = pd.to_datetime(out[col], utc=True)
    out["delay_min"] = np.where(
        out["cancelled"] | out["actual_arr"].isna(),
        np.nan,
        (out["actual_arr"] - out["sched_arr"]).dt.total_seconds() / 60.0,
    )
    return out.sort_values(["date", "sched_arr", "rid", "route_id"]).reset_index(drop=True)


def _service_class():
    from rail_risk.api.service import NotFound, RiskService

    @dataclass
    class ReplayService(RiskService):
        """The API's RiskService reading precomputed features instead of a live cache.

        `pre` maps (rid, route, prediction time) to the feature row built from the stop table as
        of that time. Everything else, including `_predict`, `risk` and `connection`, is the
        API's own code.
        """

        pre: dict[Any, dict[str, Any]] = field(default_factory=dict)

        def _features_for(self, rid, table, route_id, minutes, stale):
            hit = self.pre.get((rid, route_id, pd.Timestamp(self.now()).value))
            if hit is None:
                raise NotFound("features could not be built for that service")
            return hit, 0.0

    return ReplayService


def build_service(settings: Any, expect_run_id: str | None = None):
    """The API's RiskService around the champion by alias, with no refresher and no live store.

    Wiring follows `default_service`: the same bands, rules, feature service, connection inputs
    (alert threshold, outbound history, tails, P(miss) calibration) and the model loaded by alias.
    The store is replaced per prediction by a point-in-time one.
    """
    from pathlib import Path as _Path

    from rail_risk.api.connection_inputs import load_connection_inputs
    from rail_risk.config import load_routes
    from rail_risk.evaluation import load_bands
    from rail_risk.live import FeatureService, load_tiplocs
    from rail_risk.models import registry
    from rail_risk.repay import load_rules
    from rail_risk.spark import build_spark

    loaded = registry.load_alias(_Path("mlruns"))
    if expect_run_id and loaded.run_id != expect_run_id:
        raise ValueError(f"champion is run {loaded.run_id}, expected {expect_run_id}")
    spark = build_spark("live-eval", cores=4, driver_memory="4g", shuffle_partitions=4)
    features = FeatureService(spark, settings.data_dir / "interim", settings.routes_file)
    bundle = loaded.bundle
    cls = _service_class()
    service = cls(
        store=LiveStore(Path("."), Path("."), load_tiplocs()), features=features,
        routes=load_routes(settings.routes_file).routes,
        bands=load_bands(_Path("config/delay_bands.yaml")),
        rules=load_rules(_Path("config/delay_repay.yaml")), bundle=bundle,
        baseline=loaded.baseline, model_run_id=loaded.run_id, model_version=loaded.version,
        route_model_trained_on=set(bundle.quantile.chosen.design.categories.get("route_id", [])),
    )  # fmt: skip
    for name, value in load_connection_inputs(settings, bundle).items():
        setattr(service, name, value)
    return service


def set_time(svc: Any, store: LiveStore, t: pd.Timestamp) -> None:
    """Point the service at a point-in-time store and a clock fixed at `t`."""
    svc.store = store
    svc.now = lambda: t
    svc._outbound_cache = {}  # its entries are keyed by the day, not by the store
    svc._memo = None


def batch_features(features: Any, items: list[tuple]) -> dict[Any, dict[str, Any]]:
    """Features for (key, rid, stop table, route, minutes) items through `FeatureService.batch`.

    `batch` is keyed by rid and takes one prediction horizon, so items are grouped by minutes and
    split so that no chunk holds a rid twice.
    """
    out: dict[Any, dict[str, Any]] = {}
    by_minutes: dict[int, list[tuple]] = {}
    for item in items:
        by_minutes.setdefault(int(item[4]), []).append(item)
    for minutes, group in sorted(by_minutes.items()):
        chunks: list[dict[str, tuple]] = []
        for item in group:
            for chunk in chunks:
                if item[1] not in chunk:
                    chunk[item[1]] = item
                    break
            else:
                chunks.append({item[1]: item})
        for chunk in chunks:
            built = features.batch({rid: (it[2], it[3], minutes) for rid, it in chunk.items()})
            for rid, it in chunk.items():
                if rid in built:
                    out[it[0]] = built[rid]
    return out


# single-service replay

SINGLE_STATUS = (
    "gap", "unknown_at_T", "cancelled_at_T", "arrived_by_T", "no_features", "no_label", "scored"
)  # fmt: skip


def replay_singles(
    svc: Any,
    data: ReplayData,
    universe: pd.DataFrame,
    gaps: GapIndex,
    horizons: tuple[int, ...],
    lookback_min: int = 60,
    log: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every kept (service, route, horizon) from the state known at T.

    Returns (rows, funnel). `rows` has one line per prediction with the features, the model's
    point, 80% range, P(delay >= 15, 30, 60) and the outcome. `funnel` has one line per
    (service, route, horizon) with the reason it was or was not predicted and scored.
    """
    funnel, items, meta = [], [], {}
    for r in universe.itertuples(index=False):
        for h in horizons:
            t = r.sched_arr - pd.Timedelta(minutes=h)
            end = r.sched_arr
            if pd.notna(r.actual_msg_ts) and r.actual_msg_ts > end:
                end = r.actual_msg_ts
            base = {"rid": r.rid, "route_id": r.route_id, "date": r.date, "horizon": h, "T": t}
            pre_gap = gaps.overlaps(t - pd.Timedelta(minutes=lookback_min), t)
            if gaps.overlaps(t, end):
                funnel.append({**base, "status": "gap", "gap_before_T": pre_gap})
                continue
            store = data.store_at([r.rid], t)
            set_time(svc, store, t)
            if not store.has(r.rid):
                funnel.append({**base, "status": "unknown_at_T", "gap_before_T": pre_gap})
                continue
            if store.cancellation(r.rid)[0]:
                funnel.append({**base, "status": "cancelled_at_T", "gap_before_T": pre_gap})
                continue
            stale = svc.is_stale()
            try:
                table, route_id, dest = svc._prepare(r.rid, r.route_id, strip_live=stale)
            except LookupError:
                funnel.append({**base, "status": "no_features", "gap_before_T": pre_gap})
                continue
            if pd.notna(dest["actual_arr"]) and not stale:
                funnel.append({**base, "status": "arrived_by_T", "gap_before_T": pre_gap})
                continue
            minutes = max(1, svc._minutes(dest["sched_arr"]))
            key = (r.rid, route_id, t.value)
            items.append((key, r.rid, table, route_id, minutes))
            meta[key] = (r, h, t, stale, pre_gap, base)
    if log:
        log.info("features_start", predictions=len(items))
    built = batch_features(svc.features, items)
    rows = []
    for key, (r, h, t, stale, pre_gap, base) in meta.items():
        feat = built.get(key)
        if not feat:
            funnel.append({**base, "status": "no_features", "gap_before_T": pre_gap})
            continue
        svc.pre[key] = feat
        store = data.store_at([r.rid], t)
        set_time(svc, store, t)
        minutes = int(feat["horizon_min"])
        point, lo, hi, probs, used = svc._predict(feat, r.route_id, minutes, {}, [])
        row = {c: feat.get(c) for c in _feature_columns()}
        row.update(
            rid=r.rid, route_id=r.route_id, date=r.date, horizon=h, T=t, sched_arr=r.sched_arr,
            toc=r.toc, stale=stale, gap_before_T=pre_gap, delay_min=r.delay_min,
            point=float(point), lo=float(lo), hi=float(hi), horizon_used=int(used),
        )  # fmt: skip
        for edge in (15, 30, 60):
            row[f"p{edge}"] = float(_threshold(probs, svc.bands, edge))
        row["probs"] = [float(x) for x in probs[0]]
        rows.append(row)
        status = "scored" if not np.isnan(r.delay_min) else "no_label"
        funnel.append({**base, "status": status, "gap_before_T": pre_gap})
    return pd.DataFrame(rows), pd.DataFrame(funnel)


def _feature_columns() -> list[str]:
    from rail_risk.features.snapshots import FEATURE_COLUMNS

    return list(FEATURE_COLUMNS)


def _threshold(probs: np.ndarray, bands: Any, minutes: int) -> float:
    from rail_risk.evaluation import threshold_probability

    return float(threshold_probability(probs, bands, minutes)[0])


def parity_with_risk(
    svc: Any, data: ReplayData, rows: pd.DataFrame, sample: int, seed: int
) -> dict[str, Any]:
    """Run a sample of the replayed predictions through `RiskService.risk` itself.

    `risk` is the code behind POST /risk. Its answer is rounded for the response, so the check is
    on the median delay to 0.1 minute and P(delay >= 15) to four places.
    """
    if rows.empty:
        return {"checked": 0}
    pick = rows.sample(min(sample, len(rows)), random_state=seed)
    agree_median = agree_p15 = 0
    worst = 0.0
    for r in pick.itertuples(index=False):
        t = pd.Timestamp(r.T)
        set_time(svc, data.store_at([r.rid], t), t)
        resp = svc.risk(r.rid, None, "single", r.route_id, "replay-parity")
        same_median = (
            resp.median_delay_minutes is not None
            and abs(resp.median_delay_minutes - round(r.point, 1)) < 1e-9
        )
        same_p15 = (
            resp.p_delay_at_least_15 is not None
            and abs(resp.p_delay_at_least_15 - round(r.p15, 4)) < 1e-9
        )
        agree_median += bool(same_median)
        agree_p15 += bool(same_p15)
        if resp.median_delay_minutes is not None:
            worst = max(worst, abs(resp.median_delay_minutes - r.point))
    n = len(pick)
    return {
        "checked": n, "same_median_to_0.1_min": agree_median, "same_p15_to_4dp": agree_p15,
        "largest_median_difference_min": worst,
    }  # fmt: skip


# connection pairs


def build_live_pairs(universe: pd.DataFrame, mct: float, window: float) -> pd.DataFrame:
    """Every ast-bhm inbound with each bhm-eus outbound scheduled `mct` to `mct + window` minutes
    after it is due at the hub, the pair definition of the historical backtest.

    A pair is never built for an outbound that leaves before the minimum connection time, so no
    connection that is infeasible on paper is scored. `missed` is 1 when the inbound's actual
    arrival plus the connection time is after the outbound's actual departure, and unknown when
    either train has no actual time or was cancelled.
    """
    inbound = universe[universe["route_id"] == "ast-bhm"]
    outbound = universe[(universe["route_id"] == "bhm-eus") & universe["hub_sched_dep"].notna()]
    rows = []
    mct_td = pd.Timedelta(minutes=mct)
    for day in sorted(set(inbound["date"])):
        outs = outbound[outbound["date"] == day]
        for i in inbound[inbound["date"] == day].itertuples(index=False):
            near = outs[
                (outs["hub_sched_dep"] >= i.sched_arr - pd.Timedelta(minutes=30))
                & (outs["hub_sched_dep"] <= i.sched_arr + pd.Timedelta(hours=6))
            ]["rid"].tolist()
            for o in outs.itertuples(index=False):
                gap = (o.hub_sched_dep - i.sched_arr).total_seconds() / 60.0
                if gap < mct or gap > mct + window:
                    continue
                known = (
                    not (i.cancelled or o.cancelled)
                    and pd.notna(i.actual_arr)
                    and pd.notna(o.hub_actual_dep)
                )
                missed = float(i.actual_arr + mct_td > o.hub_actual_dep) if known else np.nan
                ends = [
                    t for t in (o.hub_sched_dep, i.actual_msg_ts, o.hub_dep_msg_ts) if pd.notna(t)
                ]
                rows.append(
                    {
                        "date": day, "inbound_rid": i.rid, "outbound_rid": o.rid,
                        "in_sched_arr": i.sched_arr, "out_sched_dep": o.hub_sched_dep,
                        "gap": gap, "missed": missed, "window_end": max(ends),
                        "outbound_candidates": near,
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


def replay_pairs(
    svc: Any,
    data: ReplayData,
    pairs: pd.DataFrame,
    gaps: GapIndex,
    ks: tuple[int, ...],
    lookback_min: int = 60,
    log: Any = None,
) -> pd.DataFrame:
    """Run `RiskService.connection` for each pair, k minutes before the outbound leaves.

    The service sees only the messages at or before T, for the inbound and for every outbound
    that could be its planned train or a later alternative. The answer is the API's own P(miss),
    with the recalibration and the alert flag it applies. Returns one line per (pair, k).
    """

    out, tasks, items = [], [], {}
    for p in pairs.itertuples(index=False):
        for k in ks:
            t = p.out_sched_dep - pd.Timedelta(minutes=k)
            rec = {
                "date": p.date, "inbound_rid": p.inbound_rid, "outbound_rid": p.outbound_rid,
                "k": k, "T": t, "gap": p.gap, "missed": p.missed,
                "gap_before_T": gaps.overlaps(t - pd.Timedelta(minutes=lookback_min), t),
            }  # fmt: skip
            if gaps.overlaps(t, p.window_end):
                out.append({**rec, "status": "gap"})
                continue
            rids = [p.inbound_rid, *p.outbound_candidates]
            store = data.store_at(rids, t)
            set_time(svc, store, t)
            if not store.has(p.inbound_rid):
                out.append({**rec, "status": "unknown_at_T"})
                continue
            stale = svc.is_stale()
            try:
                table, route_id, dest = svc._prepare(p.inbound_rid, "ast-bhm", strip_live=stale)
            except LookupError as exc:
                out.append({**rec, "status": "no_stops_at_T", "refusal": str(exc)})
                continue
            key = (p.inbound_rid, route_id, t.value)
            if not store.cancellation(p.inbound_rid)[0] and not (
                pd.notna(dest["actual_arr"]) and not stale
            ):
                minutes = max(1, svc._minutes(dest["sched_arr"]))
                items[key] = (key, p.inbound_rid, table, route_id, minutes)
            tasks.append((rec, p, key, t, rids))
    if log:
        log.info("pair_features_start", pairs=len(tasks), features=len(items))
    svc.pre.update(batch_features(svc.features, list(items.values())))
    for rec, p, _key, t, rids in tasks:
        set_time(svc, data.store_at(rids, t), t)
        hhmm = pd.Timestamp(p.out_sched_dep).tz_convert(LONDON).strftime("%H:%M")
        try:
            resp = svc.connection(p.inbound_rid, hhmm, None, "single", "replay")
        except LookupError as exc:
            out.append({**rec, "status": f"error: {exc}"})
            continue
        out.append(
            {
                **rec, "status": resp.status, "p_miss": resp.p_miss, "alert": resp.alert,
                "state": None if resp.inbound is None else resp.inbound.state,
                "same_outbound": resp.outbound is not None
                and resp.outbound.rid == p.outbound_rid,
                "refusal": resp.refusal,
            }
        )  # fmt: skip
    return pd.DataFrame(out)


def fit_miss_baseline(settings: Any, train_days: list[date], mct: float, window: float):
    """The gap-bucket miss baseline of the backtest, fitted on training dates from HSP."""
    import sqlite3
    from types import SimpleNamespace

    from rail_risk.backtest import STOP_COLUMNS, MissBaseline, build_pairs

    interim = settings.data_dir / "interim"
    stops = pd.read_parquet(interim / "_unified" / "hsp_stops.parquet", columns=STOP_COLUMNS)
    stops["_day"] = pd.to_datetime(stops["date_of_service"]).dt.date
    db = sqlite3.connect(f"file:{interim / 'hsp_manifest.sqlite'}?mode=ro", uri=True)
    try:
        eus = {r[0] for r in db.execute("SELECT rid FROM rid_routes WHERE route='bhm-eus'")}
        ast = {r[0] for r in db.execute("SELECT rid FROM rid_routes WHERE route='ast-bhm'")}
    finally:
        db.close()
    ctx = SimpleNamespace(all_stops=stops, ast_rids=ast, eus_rids=eus)
    train_pairs = build_pairs(ctx, train_days, "BHM", "EUS", mct, window)
    baseline = MissBaseline(train_pairs, mct)
    return baseline, {
        "training_days": len(train_days), "training_pairs": int(len(train_pairs)),
        "pairs_with_known_outcome": int(train_pairs["missed"].notna().sum()),
        "overall_miss_rate": baseline.overall,
        "rate_by_gap_bucket": {str(k): float(v) for k, v in sorted(baseline.rate.items())},
    }  # fmt: skip


# scoring

MIN_DATES_FOR_CI = 5
BOOTSTRAP_DEFAULT = {"resamples": 1000, "seed": 20261015, "confidence": 0.95}


def _clean(value: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, NaN and inf to None, dates to ISO text."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _drop_ci(cell: dict[str, Any]) -> dict[str, Any]:
    return {**cell, "ci_low": None, "ci_high": None}


def _bins(n: int) -> int:
    return int(min(10, max(2, n // 10)))


def fit_baselines(settings: Any, split_file: Path, bands: Any):
    """Persistence and historical baselines fitted on training dates only, as `evaluate` does.

    Returns (persistence interval baseline, historical baseline, split, training row count).
    """
    from rail_risk.evaluation import (
        HistoricalBaseline,
        PersistenceIntervalBaseline,
        complete_days,
        load_snapshots,
        load_split_config,
        make_split,
    )

    cfg = load_split_config(split_file)
    snaps = load_snapshots(settings.data_dir / "features" / "snapshots")
    snaps = snaps[snaps["route_id"].isin(cfg.routes)]
    days = complete_days(
        settings.data_dir / "interim" / "hsp_manifest.sqlite",
        sorted(snaps["route_id"].unique()),
        cfg.windows_per_day,
    )
    split = make_split([d for d in days if d in set(snaps["service_date"])], cfg)
    train = snaps[snaps["service_date"].isin(set(split.train)) & snaps["label_delay_min"].notna()]
    return (
        PersistenceIntervalBaseline(bands).fit(train),
        HistoricalBaseline(bands).fit(train),
        split,
        len(train),
    )


def _baseline_frame(sub: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "route_id": sub["route_id"].to_numpy(),
            "dest_hour": sub["dest_hour"].astype(float).to_numpy(),
            "hist_median_28d": pd.to_numeric(sub["hist_median_28d"], errors="coerce").to_numpy(),
            "horizon_min": sub["horizon"].to_numpy(),
            "last_stop_lateness_min": pd.to_numeric(
                sub["last_stop_lateness_min"], errors="coerce"
            ).to_numpy(),
        }
    )
    return df


def score_cell(
    sub: pd.DataFrame,
    persistence: Any,
    historical: Any,
    bands: Any,
    bar_name: str | None,
    boot: dict[str, Any],
    detail: bool = True,
) -> dict[str, Any]:
    """Every live metric for one route and horizon, with bootstrap CIs over service dates."""
    from rail_risk import evaluation as ev
    from rail_risk.lateness import group_labels, lateness_groups

    y = sub["delay_min"].to_numpy(dtype=float)
    dates = sub["date"].to_numpy()
    n_dates = int(len(set(dates)))
    df = _baseline_frame(sub)
    probs = np.vstack(sub["probs"].to_list())
    model = ev.Predictions(
        sub["point"].to_numpy(float), sub["lo"].to_numpy(float), sub["hi"].to_numpy(float), probs
    )
    preds = {
        "model": model,
        "timetable": ev.timetable_baseline(df, bands),
        "persistence": persistence.predict(df),
        "historical": historical.predict(df),
    }
    kw = dict(boot)
    scores = {name: ev.score(dates, y, p, bands, **kw) for name, p in preds.items()}
    ci_ok = n_dates >= MIN_DATES_FOR_CI
    out: dict[str, Any] = {
        "n_rows": int(len(y)), "n_dates": n_dates, "ci_available": ci_ok,
        "delay_ge_15_rate": float((y >= 15).mean()), "delay_ge_15_count": int((y >= 15).sum()),
        "mean_delay_min": float(y.mean()),
        "not_started_share": float(df["last_stop_lateness_min"].isna().mean()),
    }  # fmt: skip
    for name, sc in scores.items():
        keep = (
            ("mae", "coverage", "width")
            if name in ("model", "persistence", "historical")
            else ("mae",)
        )
        out[name] = {m: (sc[m] if ci_ok else _drop_ci(sc[m])) for m in keep}
    mae_live = {k: scores[k]["mae"]["value"] for k in ("timetable", "persistence", "historical")}
    best_live = min(mae_live, key=lambda k: mae_live[k])

    def paired(name: str) -> dict[str, Any]:
        d = ev.paired_mae_difference(dates, y, model.point, preds[name].point, **kw)
        return {"bar": name, **(d if ci_ok else _drop_ci(d))}

    out["vs_validation_bar"] = paired(bar_name) if bar_name else None
    out["vs_best_baseline_on_live_rows"] = paired(best_live)
    out["vs_persistence"] = paired("persistence")
    out["vs_historical"] = paired("historical")
    # P(delay >= 15): model against the historical baseline
    outcome = (y >= 15).astype(float)
    bins = _bins(len(y))
    p_model = ev.threshold_probability(probs, bands, 15)
    p_hist = ev.threshold_probability(preds["historical"].probs, bands, 15)
    rel = {
        "model": ev.reliability(dates, p_model, outcome, bins=bins, **kw),
        "historical": ev.reliability(dates, p_hist, outcome, bins=bins, **kw),
    }
    out["p_delay_ge_15"] = {
        "bins": bins,
        **{
            name: {
                "mean_predicted": float(np.mean(p)),
                "brier": r["brier"] if ci_ok else _drop_ci(r["brier"]),
                "ece": r["ece"] if ci_ok else _drop_ci(r["ece"]),
            }
            for name, p, r in (
                ("model", p_model, rel["model"]),
                ("historical", p_hist, rel["historical"]),
            )
        },
    }  # fmt: skip
    if detail:
        groups = lateness_groups(
            sub.assign(last_stop_lateness_min=df["last_stop_lateness_min"].to_numpy())
        )
        out["coverage_by_observed_lateness"] = {
            label: (cell if ci_ok else _drop_ci(cell))
            for label, cell in ev.coverage_by_group(
                dates, y, ev.Predictions(model.point, model.lo, model.hi), bands, groups,
                group_labels(), **kw,
            ).items()
        }  # fmt: skip
    return out


def daily_mae(sub: pd.DataFrame, persistence: Any, historical: Any, bands: Any, bar: str | None):
    """Model and bar MAE for each service date, for the stability table."""
    from rail_risk import evaluation as ev

    df = _baseline_frame(sub)
    preds = {
        "timetable": ev.timetable_baseline(df, bands).point,
        "persistence": persistence.predict(df).point,
        "historical": historical.predict(df).point,
    }
    y = sub["delay_min"].to_numpy(float)
    bar_point = preds[bar] if bar else preds["historical"]
    out = {}
    for day in sorted(set(sub["date"])):
        m = (sub["date"] == day).to_numpy()
        out[day] = {
            "n": int(m.sum()),
            "model_mae": float(np.abs(sub["point"].to_numpy(float)[m] - y[m]).mean()),
            "bar_mae": float(np.abs(bar_point[m] - y[m]).mean()),
        }
    return out


# the HSP spot check


def hsp_spot_check(
    check_dir: Path, day: date, universe: pd.DataFrame, data: ReplayData, routes: list[Any]
) -> dict[str, Any]:
    """Agreement of the actual arrival times of one day between HSP and the Darwin capture.

    HSP for the day was fetched into `check_dir`, a data directory of its own, so the HSP history
    behind the champion and its training split are untouched. For each route the arrival at the
    route destination is compared service by service, on the same rid, to the minute and within
    60 seconds. All stops are also compared through `actual_time_agreement`.
    """
    import sqlite3

    import pyarrow.parquet as pq

    from rail_risk.live import actual_time_agreement

    interim = check_dir / "interim"
    files = sorted((interim / "hsp_stops" / f"date={day.isoformat()}").glob("*.parquet"))
    if not files:
        return {"available": False, "reason": f"no HSP stops for {day} in {check_dir}"}
    stops = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    stops = stops.drop_duplicates(["rid", "seq"])
    db = sqlite3.connect(f"file:{interim / 'hsp_manifest.sqlite'}?mode=ro", uri=True)
    try:
        rid_routes = db.execute("SELECT rid, route FROM rid_routes").fetchall()
    finally:
        db.close()
    dest_crs = {r.id: r.destination for r in routes}
    out: dict[str, Any] = {"available": True, "day": day.isoformat(), "routes": {}}
    day_uni = universe[universe["date"] == day].set_index(["rid", "route_id"])
    pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for route_id in ("bhm-eus", "ast-bhm"):
        rids = sorted({rid for rid, route in rid_routes if route == route_id})
        cell = dict.fromkeys(
            ("hsp_services", "in_darwin_capture", "both_have_actual", "identical_to_the_minute",
             "within_60_seconds", "only_hsp_has_actual", "only_darwin_has_actual"), 0
        )  # fmt: skip
        for rid in rids:
            table = stops[stops["rid"] == rid].sort_values("seq")
            dest = table[table["location"] == dest_crs[route_id]]
            if (
                dest.empty
                or pd.isna(dest.iloc[-1]["sched_arr"])
                or table["candidate_cancelled"].any()
            ):
                continue
            cell["hsp_services"] += 1
            if (rid, route_id) not in day_uni.index:
                continue
            cell["in_darwin_capture"] += 1
            a = dest.iloc[-1]["actual_arr"]
            b = day_uni.loc[(rid, route_id), "actual_arr"]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a):
                cell["only_darwin_has_actual"] += 1
            elif pd.isna(b):
                cell["only_hsp_has_actual"] += 1
            else:
                cell["both_have_actual"] += 1
                gap = abs((pd.Timestamp(a) - pd.Timestamp(b)).total_seconds())
                cell["identical_to_the_minute"] += gap == 0
                cell["within_60_seconds"] += gap <= 60
            if rid not in pairs:
                pairs[rid] = (table, data.store_at([rid], None).stop_table(rid))
        n = cell["both_have_actual"]
        cell["identical_share"] = cell["identical_to_the_minute"] / n if n else None
        cell["within_60_seconds_share"] = cell["within_60_seconds"] / n if n else None
        out["routes"][route_id] = cell
    both = sum(c["both_have_actual"] for c in out["routes"].values())
    same = sum(c["identical_to_the_minute"] for c in out["routes"].values())
    out["destination_arrivals"] = {
        "compared": both, "identical_to_the_minute": same,
        "identical_share": same / both if both else None,
    }  # fmt: skip
    out["all_stops"] = actual_time_agreement(pairs) if pairs else None
    return out


# reference numbers of the test block

BAR_ORDER = ("timetable", "persistence", "historical")


def load_test_reference(model_dir: Path, backtest_json: Path) -> dict[str, Any]:
    """The test-block numbers of the champion run: results.json and the connection backtest."""
    import json

    results = json.loads((model_dir / "results.json").read_text(encoding="utf-8"))
    backtest = json.loads(backtest_json.read_text(encoding="utf-8"))
    return {"results": results, "backtest": backtest}


def verdict(
    live: dict[str, Any] | None,
    test: dict[str, Any] | None,
    *,
    lower_is_better: bool = True,
    target: float | None = None,
) -> str:
    """Live against test as better, similar or worse, by whether the 95% intervals overlap.

    Overlapping intervals are called similar, which with a handful of dates means only that the
    data cannot tell them apart. With a target (the nominal 80% coverage) the labels are about
    distance from it instead.
    """
    if not live or not test or live.get("value") is None or test.get("value") is None:
        return "not comparable"
    lv, tv = live["value"], test["value"]
    if live.get("ci_low") is None or test.get("ci_low") is None:
        side = "lower" if lv < tv else "higher" if lv > tv else "equal"
        return f"not testable (fewer than {MIN_DATES_FOR_CI} dates), live {side}"
    if live["ci_low"] <= test["ci_high"] and test["ci_low"] <= live["ci_high"]:
        return "similar"
    if target is not None:
        return (
            "closer to the target"
            if abs(lv - target) < abs(tv - target)
            else "further from the target"
        )
    if lower_is_better:
        return "better" if lv < tv else "worse"
    return "better" if lv > tv else "worse"


def compare_with_test(live: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """Each live single-service and connection metric next to its test-block counterpart."""
    rows = []
    results = ref["results"]["routes"]
    for route_id, horizons in live["singles"].items():
        for horizon, cell in horizons.items():
            r = results[route_id]
            test = r["results"]["model"]["test"][horizon]
            hist = r["results"]["historical"]["test"][horizon]
            bar = r["mae_vs_bar"][horizon]
            p15 = test["threshold_reliability"]["15"]
            metrics = [
                ("MAE, minutes", cell["model"]["mae"], test["mae"], {}),
                ("MAE minus validation bar", cell["vs_validation_bar"], bar, {}),
                ("80% coverage", cell["model"]["coverage"], test["coverage"], {"target": 0.8}),
                ("80% interval width", cell["model"]["width"], test["width"], {}),
                ("P(15+) Brier", cell["p_delay_ge_15"]["model"]["brier"], p15["brier"], {}),
                ("P(15+) ECE", cell["p_delay_ge_15"]["model"]["ece"], p15["ece"], {}),
                (
                    "P(15+) Brier, historical baseline",
                    cell["p_delay_ge_15"]["historical"]["brier"],
                    hist["threshold_reliability"]["15"]["brier"],
                    {},
                ),
            ]
            for name, lv, tv, extra in metrics:
                rows.append(
                    {
                        "route": route_id, "horizon": int(horizon), "metric": name,
                        "live": lv, "test": {k: tv.get(k) for k in ("value", "ci_low", "ci_high")},
                        "verdict": verdict(lv, tv, **extra),
                        "test_bar": bar["bar"] if name == "MAE minus validation bar" else None,
                        "live_bar": cell["vs_validation_bar"]["bar"]
                        if name == "MAE minus validation bar" else None,
                    }
                )  # fmt: skip
    conn = []
    scores = ref["backtest"]["scores"]["test"]
    for label, cell in live["connections"]["cells"].items():
        test_cell = scores.get(label.split(" (")[0])
        if test_cell is None:
            continue
        sub = test_cell["not_yet_arrived"] if "not yet arrived" in label else test_cell
        if sub is None:
            continue
        for name, key in (("P(miss) Brier", "brier"), ("P(miss) ECE", "ece")):
            conn.append(
                {
                    "cell": label, "metric": name, "live": cell["model"][key],
                    "test": sub["model"][key],
                    "verdict": verdict(cell["model"][key], sub["model"][key]),
                }
            )  # fmt: skip
        conn.append(
            {
                "cell": label, "metric": "P(miss) Brier, gap-bucket baseline",
                "live": cell["baseline"]["brier"], "test": sub["baseline"]["brier"],
                "verdict": verdict(cell["baseline"]["brier"], sub["baseline"]["brier"]),
            }
        )  # fmt: skip
    return {"singles": rows, "connections": conn}


# connection scoring


def score_connections(
    pred: pd.DataFrame, baseline: Any, tau: float, costs: tuple[float, float], boot: dict[str, Any]
) -> dict[str, Any]:
    """P(miss) against the gap-bucket baseline, and the alert at the frozen threshold."""
    from rail_risk.alerts import policy_stats
    from rail_risk.backtest import score_block

    ok = pred[(pred["status"] == "ok") & pred["p_miss"].notna()]
    known = ok[ok["missed"].notna()].copy()
    known["baseline"] = baseline.predict(known["gap"].to_numpy()) if len(known) else []
    known["realised_arrival"], known["lo"], known["hi"] = np.nan, np.nan, np.nan
    known["stranded"] = False

    def block(sub: pd.DataFrame) -> dict[str, Any] | None:
        if len(sub) < 20:
            return None
        cell = score_block(sub, seed=boot["seed"], resamples=boot["resamples"])
        if cell["dates"] < MIN_DATES_FOR_CI:
            for part in ("model", "baseline"):
                cell[part] = {m: _drop_ci(v) for m, v in cell[part].items()}
            cell["ci_available"] = False
        else:
            cell["ci_available"] = True
        for drop in ("reliability_model", "reliability_baseline", "interval_coverage",
                     "interval_width_min", "interval_pairs", "stranded_realised"):  # fmt: skip
            cell.pop(drop, None)
        return cell

    def alerts(sub: pd.DataFrame) -> dict[str, Any] | None:
        if sub.empty:
            return None
        return policy_stats(
            sub["p_miss"].to_numpy(), sub["missed"].to_numpy(), tau, costs[0], costs[1]
        )

    cells: dict[str, Any] = {}
    alert_cells: dict[str, Any] = {}
    t60, t15 = known[known["k"] == 60], known[known["k"] == 15]
    waiting15 = t15[t15["state"] != "arrived"]
    spec = {
        "T-60 before outbound": t60,
        "T-15 before outbound (inbound not yet arrived)": waiting15,
        "T-15 before outbound (all pairs)": t15,
    }
    for label, sub in spec.items():
        cell = block(sub)
        if cell is not None:
            if "all pairs" in label:
                cell["excluded_arrived"] = 0
            cells[label] = cell
        alert_cells[label] = alerts(sub)
    alert_cells["both forecast times pooled (all pairs)"] = alerts(known)
    by_state = {}
    for (k, state), g in known.groupby(["k", "state"]):
        by_state[f"T-{k}, {state}"] = {
            "pairs": int(len(g)), "miss_rate": float(g["missed"].mean()),
            "mean_predicted_miss": float(g["p_miss"].mean()),
        }  # fmt: skip
    return {"cells": cells, "alerts": alert_cells, "by_state": by_state, "tau": tau}


def _reasons(g: pd.DataFrame) -> dict[str, int]:
    """Why pairs were refused or not answered, with clock times and counts masked."""
    import re

    text = g.loc[g["status"] != "ok", "refusal"].dropna() if "refusal" in g else []
    masked = [re.sub(r"\d+", "N", re.sub(r"\d\d:\d\d", "HH:MM", t)) for t in text]
    return {str(k): int(v) for k, v in pd.Series(masked, dtype=object).value_counts().items()}


def pair_funnel(pred: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, g in pred.groupby("k"):
        counts = g["status"].value_counts().to_dict()
        ok = g[g["status"] == "ok"]
        out[f"T-{k}"] = {
            "feasible_pairs": int(len(g)),
            "by_status": {str(s): int(n) for s, n in counts.items()},
            "ok_with_known_outcome": int(ok["missed"].notna().sum()),
            "ok_outcome_unknown": int(ok["missed"].isna().sum()),
            "scored_not_yet_arrived": int(
                (ok["missed"].notna() & (ok["state"] != "arrived")).sum()
            ),
            "outbound_differs_from_pair": int((~ok["same_outbound"].astype(bool)).sum()),
            "reasons": _reasons(g),
            "not_answered_after_gap_rule": int((~g["status"].isin(["gap", "ok"])).sum()),
            "of_which_gap_in_prior_hour": int(
                g.loc[~g["status"].isin(["gap", "ok"]), "gap_before_T"].astype(bool).sum()
            ),
        }  # fmt: skip
    return out


# the whole evaluation


def _hash_folder(folder: Path) -> dict[str, str]:
    import hashlib

    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.glob("*"))
        if p.is_file()
    }


def _champion_state(mlruns: Path) -> dict[str, Any]:
    from rail_risk.models import registry

    _, client = registry._client(mlruns)
    v = client.get_model_version_by_alias(registry.MODEL_NAME, registry.ALIAS)
    folder = mlruns / "artifacts" / v.run_id / "artifacts" / "model"
    return {
        "run_id": v.run_id, "registry_version": str(v.version),
        "files_sha256": _hash_folder(folder) if folder.exists() else None,
    }  # fmt: skip


def run_live_eval(
    settings: Any,
    *,
    start: date,
    end: date,
    split_file: Path,
    hsp_check_dir: Path | None,
    hsp_check_day: date | None,
    expect_run_id: str | None,
    lookback_min: int = 60,
    parity_sample: int = 60,
    window: float = 20.0,
    horizons: tuple[int, ...] = (120, 60, 15),
    ks: tuple[int, ...] = (60, 15),
    log: Any = None,
) -> dict[str, Any]:
    """Replay the window and score it. Nothing is fitted except the baselines, on training dates."""
    import time

    from rail_risk.alerts import load_alert_config
    from rail_risk.evaluation import load_bands, load_split_config
    from rail_risk.live import load_tiplocs

    started = time.monotonic()
    say = log.info if log else (lambda *a, **k: None)
    mlruns = Path("mlruns")
    before = _champion_state(mlruns)
    days = [start + pd.Timedelta(days=i).to_pytimedelta() for i in range((end - start).days + 1)]
    svc = build_service(settings, expect_run_id)
    bands = load_bands(Path("config/delay_bands.yaml"))
    split_cfg = load_split_config(split_file)
    boot = {
        "resamples": split_cfg.resamples,
        "seed": split_cfg.seed,
        "confidence": split_cfg.confidence,
    }
    say("service_ready", run_id=svc.model_run_id, version=svc.model_version, tau=svc.alert_tau)

    data = load_replay_data(settings.data_dir / "interim", days, load_tiplocs())
    universe = build_universe(data, svc.routes, ("bhm-eus", "ast-bhm"), days)
    t0 = pd.Timestamp(start.isoformat(), tz="Europe/London").tz_convert("UTC")
    t1 = pd.Timestamp((end + pd.Timedelta(days=1).to_pytimedelta()).isoformat(), tz="Europe/London")
    gaps = find_gaps(data.stamps, t0, t1.tz_convert("UTC"))
    gi = GapIndex(gaps)
    say("capture_gaps", gaps=len(gaps), hours=round(float(gaps["minutes"].sum() / 60), 1))

    rows, funnel = replay_singles(svc, data, universe, gi, horizons, lookback_min, log)
    parity = parity_with_risk(svc, data, rows, parity_sample, boot["seed"])
    say("singles_done", predictions=len(rows), parity=parity)

    persistence, historical, split, n_train = fit_baselines(settings, split_file, bands)
    ref = load_test_reference(
        mlruns / "artifacts" / (expect_run_id or svc.model_run_id) / "artifacts" / "model",
        Path("reports/connection_backtest.json"),
    )
    scored = rows[rows["delay_min"].notna()]
    singles: dict[str, Any] = {}
    strict: dict[str, Any] = {}
    daily: dict[str, Any] = {}
    for route_id in ("bhm-eus", "ast-bhm"):
        for h in horizons:
            sub = scored[(scored["route_id"] == route_id) & (scored["horizon"] == h)]
            if sub.empty:
                continue
            bar = ref["results"]["routes"][route_id]["mae_vs_bar"][str(h)]["bar"]
            test_cell = ref["results"]["routes"][route_id]["results"]["model"]["test"][str(h)]
            not_started = test_cell["coverage_by_group"]["not_started"]["n"]
            singles.setdefault(route_id, {})[str(h)] = score_cell(
                sub, persistence, historical, bands, bar, boot
            )
            singles[route_id][str(h)]["test_not_started_share"] = float(
                not_started / test_cell["n_rows"]["value"]
            )
            calm = sub[~sub["gap_before_T"]]
            if len(calm) >= 20:
                strict.setdefault(route_id, {})[str(h)] = score_cell(
                    calm, persistence, historical, bands, bar, boot, detail=False
                )
                strict[route_id][str(h)]["test_mae"] = test_cell["mae"]
            daily.setdefault(route_id, {})[str(h)] = daily_mae(
                sub, persistence, historical, bands, bar
            )
    for route_id, per_h in singles.items():
        for h, cell in per_h.items():
            sub = scored[(scored["route_id"] == route_id) & (scored["horizon"] == int(h))]
            top = sub["date"].value_counts()
            cell["largest_date"] = {"date": top.index[0], "share": float(top.iloc[0] / len(sub))}
    hist_by_day = (
        rows.groupby(["route_id", "date"])["hist_n_28d"]
        .apply(lambda s: float(pd.to_numeric(s, errors="coerce").mean()))
        .unstack(0)
        .to_dict()
    )

    # connections
    mct = float(svc.stations["BHM"].minutes)
    pairs = build_live_pairs(universe, mct, window)
    pair_pred = replay_pairs(svc, data, pairs, gi, ks, lookback_min, log)
    baseline, baseline_info = fit_miss_baseline(settings, list(split.train), mct, window)
    alert_cfg = load_alert_config(Path("config/alerts.yaml"))
    conn = score_connections(
        pair_pred,
        baseline,
        float(svc.alert_tau),
        (alert_cfg.false_alert, alert_cfg.missed_alert),
        boot,
    )
    conn["pair_counts"] = {
        "feasible_pairs": int(len(pairs)),
        "feasible_pairs_known_outcome": int(pairs["missed"].notna().sum()),
        "by_forecast": pair_funnel(pair_pred),
        "min_connection_time": mct, "window_minutes": window,
        "miss_rate_all_known_pairs": float(pairs["missed"].dropna().mean()),
    }  # fmt: skip
    conn["baseline"] = baseline_info
    conn["strict_pre_T"] = {
        label: block_strict
        for label, block_strict in _strict_pairs(pair_pred, baseline, conn["tau"], boot).items()
    }

    hsp = (
        hsp_spot_check(hsp_check_dir, hsp_check_day, universe, data, svc.routes)
        if hsp_check_dir and hsp_check_day
        else {"available": False, "reason": "not requested"}
    )
    after = _champion_state(mlruns)
    n_services = int(universe.groupby("route_id").size().sum())
    result: dict[str, Any] = {
        "run_id": svc.model_run_id,
        "registry_version": svc.model_version,
        "window": {"start": start, "end": end, "service_dates": days},
        "champion": {
            "before": before,
            "after": after,
            "unchanged": before == after and before["run_id"] == svc.model_run_id,
        },
        "frozen_inputs": {
            "alert_tau": svc.alert_tau,
            "pmiss_calibration": svc.pmiss_calibration.method,
            "baseline_training_days": [split.train[0], split.train[-1]],
            "baseline_training_day_count": len(split.train),
            "baseline_training_rows": n_train,
        },
        "bootstrap": boot,
        "capture": _capture_summary(gaps, days, data),
        "services": _service_summary(universe, funnel, horizons),
        "parity_with_risk": parity,
        "hsp_spot_check": hsp,
        "singles": singles,
        "singles_strict_pre_T": strict,
        "daily_mae": daily,
        "hist_n_28d_by_day": hist_by_day,
        "connections": conn,
        "lookback_minutes": lookback_min,
        "seconds": round(time.monotonic() - started),
        "n_services_in_universe": n_services,
    }
    result["comparison_with_test"] = compare_with_test(result, ref)
    return result


def _strict_pairs(pred: pd.DataFrame, baseline: Any, tau: float, boot: dict[str, Any]):
    """The pair scores again without the pairs that had a gap in the hour before T."""
    calm = pred[~pred["gap_before_T"].astype(bool)]
    sc = score_connections(calm, baseline, tau, (1.0, 5.0), boot)
    return {label: cell for label, cell in sc["cells"].items()}


def _capture_summary(gaps: pd.DataFrame, days: list[date], data: ReplayData) -> dict[str, Any]:
    total = gap_hours_by_day(gaps, days)
    service = gap_hours_by_day(gaps, days, (5, 24))
    return {
        "max_gap_minutes": MAX_GAP.total_seconds() / 60.0,
        "message_times_used": int(len(data.stamps)),
        "gaps": int(len(gaps)),
        "gap_hours_total": float(gaps["minutes"].sum() / 60.0),
        "longest_gap_minutes": float(gaps["minutes"].max()) if len(gaps) else 0.0,
        "hours_missing_by_day": {d.isoformat(): total[d] for d in days},
        "hours_missing_by_day_05_to_24": {d.isoformat(): service[d] for d in days},
        "gap_list": [
            {"start": g.gap_start, "end": g.gap_end, "minutes": g.minutes}
            for g in gaps.itertuples()
        ],
    }  # fmt: skip


def _service_summary(
    universe: pd.DataFrame, funnel: pd.DataFrame, horizons: tuple[int, ...]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for route_id in ("bhm-eus", "ast-bhm"):
        total = int((universe["route_id"] == route_id).sum())
        by_day = universe[universe["route_id"] == route_id].groupby("date").size()
        cell: dict[str, Any] = {
            "services_in_window": total,
            "services_by_day": {d.isoformat(): int(n) for d, n in by_day.items()},
        }
        for h in horizons:
            f = funnel[(funnel["route_id"] == route_id) & (funnel["horizon"] == h)]
            counts = f["status"].value_counts().to_dict()
            gap_days = f[f["status"] == "gap"].groupby("date").size()
            cell[str(h)] = {
                "services": int(len(f)),
                "kept_after_gap_exclusion": int((f["status"] != "gap").sum()),
                "excluded_for_gap": int(counts.get("gap", 0)),
                **{s: int(counts.get(s, 0)) for s in SINGLE_STATUS if s != "gap"},
                "excluded_for_gap_by_day": {d.isoformat(): int(n) for d, n in gap_days.items()},
                "kept_by_day": {
                    d.isoformat(): int(n)
                    for d, n in f[f["status"] != "gap"].groupby("date").size().items()
                },
            }
        out[route_id] = cell
    return out


# reports


def _f(cell: dict[str, Any] | None, digits: int = 3) -> str:
    """'value (low to high)', or just the value when there is no interval."""
    if not cell or cell.get("value") is None:
        return "n/a"
    text = f"{cell['value']:.{digits}f}"
    if cell.get("ci_low") is None or cell.get("ci_high") is None:
        return text
    return f"{text} ({cell['ci_low']:.{digits}f} to {cell['ci_high']:.{digits}f})"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def render_markdown(r: dict[str, Any]) -> str:
    w = r["window"]
    cap = r["capture"]
    days = [d if isinstance(d, str) else d.isoformat() for d in w["service_dates"]]
    horizons = sorted({int(h) for route in r["singles"].values() for h in route}, reverse=True)
    out: list[str] = ["# Live evaluation", ""]
    out.append(
        f"The champion model, MLflow run {r['run_id']} (registry version {r['registry_version']}), "
        f"was scored on service dates {days[0]} to {days[-1]} of captured Darwin data, routes "
        "bhm-eus and ast-bhm. The model stayed frozen: it was trained and calibrated on dates up "
        "to 2026-10-15, and no model, recalibration or threshold was changed for this "
        f"evaluation. The alert threshold is the frozen tau of "
        f"{r['frozen_inputs']['alert_tau']:.2f}. The model files and the registry alias were "
        f"{'unchanged' if r['champion']['unchanged'] else 'CHANGED'} "
        "before and after the run (SHA-256 of every file in the model folder)."
    )
    out += ["", "## Method", ""]
    out.append(
        "The API did not log predictions during the week, so every prediction was rebuilt after "
        "the fact as a point-in-time replay. For a service and a prediction time T, the live "
        "state was rebuilt from the Darwin messages whose message timestamp is at or before T, "
        "and no others: the events, the schedule versions (so a cancellation that arrived after "
        "T is not known at T), and the age of the newest message. That is stricter than the HSP "
        "rule used in training, which keeps events whose actual time is at or before T. The "
        "stop table, the features and the prediction come from the code the API runs: "
        "`LiveStore.stop_table`, `FeatureService.batch` (the training `build_snapshots`), "
        "`RiskService._predict` and, for connections, `RiskService.connection`. Features were "
        "built at 120, 60 and 15 minutes before the scheduled arrival. The 28 day history "
        "features come from the HSP history up to 2026-10-15, as they do in the API, so for "
        "later dates the history window holds fewer days (mean history rows per day are in the "
        "JSON). Actual arrival times come from Darwin. Message timestamps are Darwin message "
        "times, not the time the capture received the message."
    )
    n_param = r["parity_with_risk"]
    if n_param.get("checked"):
        out.append(
            f"As a check on the replay, {n_param['checked']} replayed predictions were also run "
            f"through `RiskService.risk`, the code behind POST /risk: "
            f"{n_param['same_median_to_0.1_min']} of {n_param['checked']} matched on the median "
            f"delay to 0.1 minute and {n_param['same_p15_to_4dp']} of {n_param['checked']} on "
            "P(delay of at least 15) to four places."
        )
    out += ["", "## Capture gaps", ""]
    out.append(
        f"A gap is any stretch of message time longer than {cap['max_gap_minutes']:.0f} minutes "
        f"with no Darwin message. Over the window there were {cap['gaps']} gaps totalling "
        f"{cap['gap_hours_total']:.1f} hours, the longest {cap['longest_gap_minutes']:.0f} "
        f"minutes, found from {cap['message_times_used']:,} distinct message times. "
        f"{sum(1 for g in cap['gap_list'] if g['minutes'] >= 20)} of the gaps are 20 minutes or "
        "longer. Several hourly raw files are missing altogether. Hours missing per local day:"
    )
    out.append("")
    rows = [
        [
            d,
            f"{cap['hours_missing_by_day'][d]:.1f}",
            f"{cap['hours_missing_by_day_05_to_24'][d]:.1f}",
        ]
        for d in days
    ]
    rows.append(
        ["Total", f"{sum(cap['hours_missing_by_day'].values()):.1f}",
         f"{sum(cap['hours_missing_by_day_05_to_24'].values()):.1f}"]
    )  # fmt: skip
    out.append(
        _table(["Service date", "Hours missing, whole day", "Hours missing, 05:00 to 24:00"], rows)
    )
    out += ["", "## Services kept and excluded", ""]
    out.append(
        "A service is excluded at a horizon when the window from T to its arrival overlaps a gap. "
        "The window ends at the scheduled arrival, or at the message that carried the actual "
        "arrival when that is later. The counts are per (service, route, horizon), and a service "
        "that calls at both AST, BHM and EUS counts once under each route, as in training. After "
        "the gap rule, a service is also left out when it was not yet known at T, was already "
        "cancelled or already arrived at T, or has no actual arrival (cancelled later, or the "
        "arrival was never captured)."
    )
    out.append("")
    rows = []
    for route_id, cell in r["services"].items():
        for h in horizons:
            c = cell[str(h)]
            rows.append(
                [
                    route_id, f"T-{h}", c["services"], c["excluded_for_gap"],
                    c["kept_after_gap_exclusion"], c["unknown_at_T"], c["cancelled_at_T"],
                    c["arrived_by_T"], c["no_label"], c["scored"],
                ]
            )  # fmt: skip
    out.append(
        _table(
            [
                "Route",
                "Horizon",
                "Services",
                "Excluded for gap",
                "Kept",
                "Unknown at T",
                "Cancelled at T",
                "Arrived by T",
                "No actual arrival",
                "Scored",
            ],
            rows,
        )  # fmt: skip
    )
    out += ["", "Services scored by service date (kept and with an actual arrival):", ""]
    header = ["Service date"] + [f"{rt} T-{h}" for rt in r["services"] for h in horizons]
    rows = []
    for d in days:
        row = [d]
        for rt in r["services"]:
            for h in horizons:
                row.append(r["daily_mae"].get(rt, {}).get(str(h), {}).get(d, {}).get("n", 0))
        rows.append(row)
    out.append(_table(header, rows))

    hsp = r["hsp_spot_check"]
    out += ["", "## Check of the actual times against HSP", ""]
    if hsp.get("available"):
        dest = hsp["destination_arrivals"]
        out.append(
            f"HSP for {hsp['day']} was fetched for both routes into a separate data directory and "
            "compared with the Darwin actual arrival at the route destination, service by service "
            f"on the same rid. Of {dest['compared']} arrivals that both feeds recorded, "
            f"{dest['identical_to_the_minute']} were identical to the minute "
            f"({_pct(dest['identical_share'])})."
        )
        rows = []
        for route_id, c in hsp["routes"].items():
            rows.append(
                [route_id, c["hsp_services"], c["in_darwin_capture"], c["both_have_actual"],
                 _pct(c["identical_share"]), _pct(c["within_60_seconds_share"]),
                 c["only_hsp_has_actual"], c["only_darwin_has_actual"]]
            )  # fmt: skip
        out += [
            "",
            _table(
                [
                    "Route",
                    "HSP services",
                    "Also in capture",
                    "Both have actual",
                    "Identical to the minute",
                    "Within 60 s",
                    "Actual only in HSP",
                    "Actual only in Darwin",
                ],
                rows,
            ),
        ]
        stops = hsp.get("all_stops")
        if stops:
            out.append("")
            out.append(
                f"Across every stop of the services in both feeds, "
                f"{stops['compared_actuals']} actual "
                f"times were compared: {_pct(stops['identical_share'])} identical and "
                f"{_pct(stops['within_tolerance_share'])} within {stops['tolerance_seconds']:.0f} "
                f"seconds. {stops['actual_in_hsp_only']} had an actual only in HSP and "
                f"{stops['actual_in_darwin_only']} only in Darwin."
            )
    else:
        out.append(f"Not run: {hsp.get('reason')}.")

    out += ["", "## Arrival ranges and probabilities by route and horizon", ""]
    out.append(
        "MAE is in minutes against the actual arrival delay. The validation bar is the baseline "
        "with the lowest MAE on the validation block of the champion run, the same bar used in "
        "reports/model_results.md. Paired differences are the model's MAE minus the baseline's, "
        "so a negative number favours the model. Intervals are 95% bootstrap intervals over "
        "service dates; with fewer than five dates no interval is given, because resampling so "
        "few dates says nothing. The window holds at most seven service dates, and on the days "
        "with the most captured time one date can hold most of the rows, so every interval "
        "is wide."
    )
    out.append("")
    rows = []
    for route_id, per_h in r["singles"].items():
        for h in horizons:
            c = per_h.get(str(h))
            if c is None:
                continue
            bar = c["vs_validation_bar"]
            best = c["vs_best_baseline_on_live_rows"]
            rows.append(
                [route_id, f"T-{h}", c["n_rows"], c["n_dates"], _f(c["model"]["mae"], 2),
                 bar["bar"], _f(c[bar["bar"]]["mae"], 2), _f(bar, 2),
                 f"{best['bar']}: {_f(best, 2)}"]
            )  # fmt: skip
    out.append(
        _table(
            [
                "Route",
                "Horizon",
                "Rows",
                "Dates",
                "Model MAE",
                "Validation bar",
                "Bar MAE",
                "Model minus bar",
                "Against the best baseline on these rows",
            ],
            rows,
        )
    )
    out += ["", "Nominal 80% interval, coverage and mean width in minutes:", ""]
    rows = []
    for route_id, per_h in r["singles"].items():
        for h in horizons:
            c = per_h.get(str(h))
            if c is None:
                continue
            rows.append(
                [route_id, f"T-{h}", _f(c["model"]["coverage"], 3), _f(c["model"]["width"], 1),
                 _f(c["persistence"]["coverage"], 3), _f(c["historical"]["coverage"], 3)]
            )  # fmt: skip
    out.append(
        _table(
            [
                "Route",
                "Horizon",
                "Model coverage",
                "Model width",
                "Persistence coverage",
                "Historical coverage",
            ],
            rows,
        )
    )
    out += ["", "P(delay of at least 15 minutes), model against the historical baseline:", ""]
    rows = []
    for route_id, per_h in r["singles"].items():
        for h in horizons:
            c = per_h.get(str(h))
            if c is None:
                continue
            p = c["p_delay_ge_15"]
            rows.append(
                [route_id, f"T-{h}", f"{c['delay_ge_15_count']} of {c['n_rows']}",
                 _f(p["model"]["brier"], 4), _f(p["historical"]["brier"], 4),
                 _f(p["model"]["ece"], 4), _f(p["historical"]["ece"], 4), p["bins"]]
            )  # fmt: skip
    out.append(
        _table(
            [
                "Route",
                "Horizon",
                "Delays of 15 or more",
                "Model Brier",
                "Historical Brier",
                "Model ECE",
                "Historical ECE",
                "ECE bins",
            ],
            rows,
        )
    )
    out.append("")
    out.append(
        "ECE uses equal-count bins, ten where there are enough rows and fewer otherwise. With so "
        "few "
        "delays of 15 or more, ECE and Brier for this event rest on a small number of events."
    )
    out += ["", "Coverage of the model's 80% interval by observed lateness at T:", ""]
    labels = ["not_started", "0-4", "5-14", "15+"]
    rows = []
    for route_id, per_h in r["singles"].items():
        for h in horizons:
            c = per_h.get(str(h))
            if c is None:
                continue
            cells = []
            for lab in labels:
                g = c["coverage_by_observed_lateness"].get(lab)
                if not g or not g["n"]:
                    cells.append("n/a (n=0)")
                elif g["n"] < 30:
                    cells.append(f"{g['value']:.2f} (n={int(g['n'])}, too few for an interval)")
                else:
                    cells.append(f"{_f(g, 2)} (n={int(g['n'])})")
            rows.append([route_id, f"T-{h}", *cells])
    out.append(_table(["Route", "Horizon", *labels], rows))

    out += ["", "## Day by day MAE", ""]
    out.append("Model MAE with the validation bar's MAE in brackets, and the number of services.")
    out.append("")
    header = ["Service date"] + [f"{rt} T-{h}" for rt in r["daily_mae"] for h in horizons]
    rows = []
    for d in days:
        row = [d]
        for rt in r["daily_mae"]:
            for h in horizons:
                e = r["daily_mae"][rt].get(str(h), {}).get(d)
                row.append(
                    "none" if not e else f"{e['model_mae']:.2f} ({e['bar_mae']:.2f}), n={e['n']}"
                )
        rows.append(row)
    out.append(_table(header, rows))

    conn = r["connections"]
    pc = conn["pair_counts"]
    out += ["", "## Connections: AST to BHM inbound, BHM to EUS outbound", ""]
    out.append(
        "A pair is one AST to BHM service with one BHM to EUS service scheduled "
        f"{pc['min_connection_time']:.0f} to "
        f"{pc['min_connection_time'] + pc['window_minutes']:.0f} minutes after it is due at "
        "Birmingham, the pair "
        "definition of the historical backtest, so no pair is infeasible on paper. The minimum "
        "connection time is assumed, not published. There were "
        f"{pc['feasible_pairs']} such pairs in the window, {pc['feasible_pairs_known_outcome']} "
        f"with a known outcome (both trains ran and have an actual time), and a miss rate of "
        f"{_pct(pc['miss_rate_all_known_pairs'])} among them. Each forecast is the API's "
        "`RiskService.connection` answer with the state known k minutes before the outbound "
        "leaves: the recalibration setting, the outbound delay history, the tail model and the "
        "alert flag are those the API uses. The baseline is the historical miss frequency by "
        f"scheduled gap bucket, fitted on {conn['baseline']['training_pairs']} pairs from "
        f"{conn['baseline']['training_days']} training days only."
    )
    out.append("")
    rows = []
    for label, c in pc["by_forecast"].items():
        by = c["by_status"]
        rows.append(
            [label, c["feasible_pairs"], by.get("gap", 0), by.get("ok", 0),
             sum(v for s, v in by.items() if s not in ("gap", "ok")), c["ok_with_known_outcome"],
             c["scored_not_yet_arrived"]]
        )  # fmt: skip
    for label, c in pc["by_forecast"].items():
        if c.get("reasons"):
            detail = "; ".join(f"{n} x {t}" for t, n in c["reasons"].items())
            out.append(
                f"Why pairs were refused or not answered at {label}: {detail}. Of the "
                f"{c['not_answered_after_gap_rule']} such pairs, {c['of_which_gap_in_prior_hour']} "
                f"had a gap in the {r['lookback_minutes']} minutes before T."
            )
            out.append("")
    out.append(
        _table(
            [
                "Forecast",
                "Feasible pairs",
                "Excluded for gap",
                "Answered",
                "Refused or other",
                "Answered with known outcome",
                "Of those, inbound not yet arrived",
            ],
            rows,
        )
    )
    out.append("")
    rows = []
    for label, c in conn["cells"].items():
        rows.append(
            [label, c["pairs"], _pct(c["miss_rate"]), _pct(c["mean_predicted_miss"]),
             _f(c["model"]["brier"], 4), _f(c["baseline"]["brier"], 4),
             _f(c["model"]["ece"], 4), _f(c["baseline"]["ece"], 4)]
        )  # fmt: skip
    out.append(
        _table(
            [
                "Forecast",
                "Pairs",
                "Realised miss rate",
                "Mean predicted",
                "Model Brier",
                "Baseline Brier",
                "Model ECE",
                "Baseline ECE",
            ],
            rows,
        )
    )
    out.append("")
    out.append(
        "The all-pairs T-15 row includes pairs whose inbound had already arrived, for which the "
        "delay "
        "is known, so it overstates what the model adds; the not-yet-arrived row is the fair test."
    )
    out += [
        "",
        f"Alert at the frozen tau of {conn['tau']:.2f} (false alert cost 1, missed alert cost 5, "
        "both assumed):",
        "",
    ]
    rows = []
    for label, a in conn["alerts"].items():
        if not a:
            continue
        rows.append(
            [label, a["n"], a["positives"], a["alerts"], a["true_alerts"],
             "n/a" if a["precision"] is None else f"{a['precision']:.3f}",
             "n/a" if a["recall"] is None else f"{a['recall']:.3f}"]
        )  # fmt: skip
    out.append(
        _table(["Pairs", "Count", "Misses", "Alerts", "True alerts", "Precision", "Recall"], rows)
    )

    out += ["", "## Live against the test block", ""]
    out.append(
        "Each live number next to the test-block number of the same champion run, from "
        "reports/model_results.md and the run's results.json. 'Similar' means the 95% intervals "
        "overlap, which with seven or fewer dates in each block says only that the data cannot "
        "separate them. 'Better' and 'worse' mean the intervals do not overlap. For coverage the "
        "labels are about distance from the nominal 80%."
    )
    cmp_ = r["comparison_with_test"]
    out.append("")
    rows = []
    for e in cmp_["singles"]:
        rows.append(
            [
                e["route"],
                f"T-{e['horizon']}",
                e["metric"],
                _f(e["live"], 3),
                _f(e["test"], 3),
                e["verdict"],
            ]
        )
    out.append(_table(["Route", "Horizon", "Metric", "Live", "Test", "Live against test"], rows))
    out.append("")
    rows = [
        [e["cell"], e["metric"], _f(e["live"], 4), _f(e["test"], 4), e["verdict"]]
        for e in cmp_["connections"]
    ]
    out.append(_table(["Connection forecast", "Metric", "Live", "Test", "Live against test"], rows))
    tally: dict[str, int] = {}
    for e in [*cmp_["singles"], *cmp_["connections"]]:
        tally[e["verdict"]] = tally.get(e["verdict"], 0) + 1
    out.append("")
    out.append(
        "Verdicts over all comparisons above: "
        + ", ".join(f"{n} {v}" for v, n in sorted(tally.items(), key=lambda kv: -kv[1]))
        + "."
    )

    out += ["", "## Sensitivity: no gap in the hour before T", ""]
    out.append(
        "The gap rule above looks only at the window from T to arrival. A gap that ended shortly "
        "before T can still have hidden the messages about the stops a train had already "
        "passed, so a train that was running can look not started in the replayed state. The "
        "share of predictions with no last stop known is shown for the main set, for this "
        "stricter set and for the test block. The stricter set keeps only predictions with no "
        f"gap in the {r['lookback_minutes']} minutes before T. Both sets are reported; the "
        "stricter one is a check on the main numbers."
    )
    out.append("")
    rows = []
    for route_id, per_h in r["singles_strict_pre_T"].items():
        for h in horizons:
            c = per_h.get(str(h))
            if c is None:
                continue
            main = r["singles"][route_id][str(h)]
            bar = c["vs_validation_bar"]
            rows.append(
                [route_id, f"T-{h}", f"{c['n_rows']} of {main['n_rows']}", c["n_dates"],
                 f"{_pct(main['not_started_share'], 0)} / {_pct(c['not_started_share'], 0)} / "
                 f"{_pct(main['test_not_started_share'], 0)}",
                 _f(c["model"]["mae"], 2), _f(c["test_mae"], 2),
                 verdict(c["model"]["mae"], c["test_mae"]), _f(bar, 2),
                 _f(c["model"]["coverage"], 3), _f(c["p_delay_ge_15"]["model"]["brier"], 4)]
            )  # fmt: skip
    out.append(
        _table(
            [
                "Route",
                "Horizon",
                "Rows kept of main",
                "Dates",
                "Not started: main / stricter / test",
                "Model MAE",
                "Test MAE",
                "MAE, live against test",
                "Model minus bar",
                "Coverage",
                "P(15+) Brier",
            ],
            rows,
        )
        if rows
        else "Too few rows."
    )
    strict_conn = r["connections"].get("strict_pre_T") or {}
    if strict_conn:
        out += [
            "",
            "The same check for the connection forecasts, keeping pairs with no gap in the "
            "hour before T:",
            "",
        ]
        rows = []
        for label, c in strict_conn.items():
            rows.append(
                [label, c["pairs"], _pct(c["miss_rate"]), _pct(c["mean_predicted_miss"]),
                 _f(c["model"]["brier"], 4), _f(c["baseline"]["brier"], 4),
                 _f(c["model"]["ece"], 4), _f(c["baseline"]["ece"], 4)]
            )  # fmt: skip
        out.append(
            _table(
                [
                    "Forecast",
                    "Pairs",
                    "Realised miss rate",
                    "Mean predicted",
                    "Model Brier",
                    "Baseline Brier",
                    "Model ECE",
                    "Baseline ECE",
                ],
                rows,
            )
        )
    out += ["", "## Limits of this evaluation", ""]
    shares = [
        c["largest_date"] for per_h in r["singles"].values() for c in per_h.values()
        if c.get("largest_date")
    ]  # fmt: skip
    if shares:
        top = max(shares, key=lambda x: x["share"])
        low = min(shares, key=lambda x: x["share"])
        out.append(
            "Scored predictions are concentrated on the dates with the least capture loss. In the "
            f"six route and horizon cells the busiest single date holds between "
            f"{_pct(low['share'], 0)} and {_pct(top['share'], 0)} of the scored rows."
        )
        out.append("")
    out.append(
        f"The capture lost {cap['gap_hours_total']:.1f} hours of message time in the window, so "
        "the "
        "services kept are those that happened to fall in captured stretches, mostly on the days "
        "with the fewest gaps. The window is at most seven service dates, in autumn, so there is "
        "no winter or storm season, and bootstrap intervals over so few dates are wide. The "
        "history features come from HSP up to 2026-10-15 and do not include the Darwin days of "
        "the window, as in the API. The connection pairs use the assumed 10 minute connection "
        "time. The replayed state uses Darwin message times and cannot say how late the live "
        "service would have received a message. Nothing here explains why live differs from "
        "test where it does."
    )
    text = "\n".join(out) + "\n"
    for bad in ("\u2014", "\u2013"):
        text = text.replace(bad, ", ")
    return text


def write_reports(result: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "live_eval.md"
    js = out_dir / "live_eval.json"
    js.write_text(json.dumps(_clean(result), indent=2) + "\n", encoding="utf-8")
    md.write_text(render_markdown(_clean(result)), encoding="utf-8")
    return md, js
