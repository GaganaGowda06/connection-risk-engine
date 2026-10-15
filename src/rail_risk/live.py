"""Live stop state from Darwin, and serving features from the training feature code.

The state of a service comes from the parsed Darwin tables (closed hours) plus a read-only,
tolerant read of raw hourly files that have not been parsed yet, including the hour still being
written. Messages are deduplicated on (partition, offset). A stop table in the same schema as the
HSP stops is handed to `build_snapshots`, the function that built the training features, so a
serving feature and a training feature are produced by the same code.
"""

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from rail_risk.ingest.darwin import Parsed, ReadStats, iter_lines, normalise, parse_payload

EVENT_COLUMNS = [
    "rid", "ssd", "tiploc", "event_type", "time_kind", "time", "planned_time", "late_reason",
    "message_ts", "partition", "offset",
]  # fmt: skip
DEDUP_KEY = ["partition", "offset", "rid", "tiploc", "event_type"]
STOP_COLUMNS = [
    "rid", "date_of_service", "toc_code", "seq", "location", "sched_arr", "sched_dep",
    "actual_arr", "actual_dep", "late_canc_reason", "candidate_cancelled",
]  # fmt: skip


def read_parts(directory: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Concatenate the Parquet parts of a directory one file at a time.

    Parts are written separately, so an all-null column can have a different type in each, which
    a dataset read cannot merge.
    """
    frames = []
    for path in sorted(directory.glob("*.parquet")):
        have = pq.read_schema(path).names
        frames.append(
            pq.read_table(path, columns=[c for c in (columns or have) if c in have]).to_pandas()
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_tiplocs(path: Path | str = "config/tiplocs.yaml") -> dict[str, str]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in raw["tiplocs"].items()}


@dataclass
class RefreshStats:
    parsed_rows: int = 0
    tail_files: int = 0
    tail_new_lines: int = 0
    tail_events: int = 0
    seconds: float = 0.0


@dataclass
class LiveStore:
    """In-memory Darwin state for the services that call at the watched TIPLOCs."""

    interim: Path
    raw: Path
    tiplocs: dict[str, str]
    days_back: int = 1
    events: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=EVENT_COLUMNS))
    schedules: dict[str, dict[str, Any]] = field(default_factory=dict)
    newest_message_ts: datetime | None = None
    _lines_done: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _by_rid: dict[str, pd.DataFrame] | None = None  # rebuilt after events change

    # reading

    def _parsed_keys(self) -> set[str]:
        path = self.interim / "darwin_manifest.sqlite"
        if not path.exists():
            return set()
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return {r[0] for r in db.execute("SELECT key FROM files")}
        finally:
            db.close()

    def _load_parsed(self, days: list[date]) -> int:
        """Events and schedules for watched services from the parsed Parquet tables."""
        total = 0
        watch = list(self.tiplocs)
        for day in days:
            events_dir = self.interim / "darwin" / "darwin_events" / f"date={day.isoformat()}"
            if not events_dir.exists():
                continue
            df = read_parts(events_dir, [c for c in EVENT_COLUMNS if c != "ssd"])
            if df.empty:
                continue
            rids = set(df.loc[df["tiploc"].isin(watch), "rid"])
            df = df[df["rid"].isin(rids)].assign(ssd=pd.Timestamp(day))
            self._append_events(df[EVENT_COLUMNS])
            total += len(df)
            sched_dir = self.interim / "darwin" / "darwin_schedules" / f"date={day.isoformat()}"
            if sched_dir.exists():
                sched = read_parts(sched_dir)
                for row in sched[sched["rid"].isin(rids)].to_dict("records"):
                    self._keep_schedule(row)
        return total

    def _keep_schedule(self, row: dict[str, Any]) -> None:
        old = self.schedules.get(row["rid"])
        if old is None or (row["message_ts"], row["offset"]) >= (old["message_ts"], old["offset"]):
            self.schedules[row["rid"]] = row

    def _append_events(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        merged = pd.concat([self.events, df], ignore_index=True) if len(self.events) else df.copy()
        self.events = merged.drop_duplicates(DEDUP_KEY, keep="last").reset_index(drop=True)
        self._by_rid = None

    def _read_tail(self, path: Path, key: str, stats: RefreshStats) -> None:
        """Parse the lines of a raw hourly file not yet seen. Read-only and tolerant of a
        partial final line, as the capture may be appending to this file."""
        read = ReadStats()
        out = Parsed()
        seen = self._lines_done.get(key, 0)
        n = 0
        for line in iter_lines(path, read):
            n += 1
            if n <= seen or not line.strip():
                continue
            try:
                partition, offset, kafka_ts, payload, _ = normalise(line)
            except (ValueError, KeyError, TypeError):
                continue
            if payload is None:
                continue
            parse_payload(payload, partition, offset, kafka_ts, out)
        self._lines_done[key] = n
        stats.tail_new_lines += max(0, n - seen)
        frame = pd.DataFrame(out.events)
        if not frame.empty:
            rids = set(frame.loc[frame["tiploc"].isin(list(self.tiplocs)), "rid"])
            keep = frame[frame["rid"].isin(rids)].copy()
            keep["ssd"] = pd.to_datetime(keep["ssd"])
            self._append_events(keep[EVENT_COLUMNS])
            stats.tail_events += len(keep)
            stamps = frame["message_ts"]
            newest = stamps.max().to_pydatetime()
            if self.newest_message_ts is None or newest > self.newest_message_ts:
                self.newest_message_ts = newest
        known = set(self.events["rid"])
        for rid, row in out.schedules.items():
            if rid in known:
                self._keep_schedule(row)

    def refresh(self, today: date | None = None) -> RefreshStats:
        """Reload parsed tables on first call, then read any raw hours not yet parsed."""
        import time

        started = time.monotonic()
        with self._lock:
            stats = RefreshStats()
            today = today or datetime.now(UTC).date()
            days = [date.fromordinal(today.toordinal() - k) for k in range(self.days_back + 1)]
            if not self._lines_done and self.events.empty:
                stats.parsed_rows = self._load_parsed(days)
                if len(self.events):
                    self.newest_message_ts = self.events["message_ts"].max().to_pydatetime()
            parsed = self._parsed_keys()
            for day in days:
                folder = self.raw / day.isoformat()
                if not folder.exists():
                    continue
                for path in sorted(folder.glob("*.jsonl.gz")):
                    key = f"{day.isoformat()}/{path.name.split('.')[0]}"
                    if key in parsed:
                        continue
                    stats.tail_files += 1
                    self._read_tail(path, key, stats)
            stats.seconds = time.monotonic() - started
            return stats

    # state

    def data_age_minutes(self, now: datetime) -> float | None:
        if self.newest_message_ts is None:
            return None
        return max(0.0, (now - self.newest_message_ts).total_seconds() / 60.0)

    def _events_of(self, rid: str) -> pd.DataFrame:
        """Events of one service, from an index rebuilt only when the events change."""
        index = self._by_rid
        if index is None:
            index = {r: g for r, g in self.events.groupby("rid", sort=False)}
            self._by_rid = index
        return index.get(rid, self.events.iloc[0:0])

    def has(self, rid: str) -> bool:
        return rid in self.schedules or not self._events_of(rid).empty

    def stop_table(self, rid: str) -> pd.DataFrame:
        """The service's calling points in the HSP stop schema, from the latest Darwin messages.

        Forecasts are not actuals: an actual time exists only where Darwin reported one. Passing
        points are dropped. A location with a mapped CRS code uses it, others keep the TIPLOC.
        """
        ev = self._events_of(rid)
        ev = ev[ev["event_type"].isin(["arr", "dep"])]
        if ev.empty:
            return pd.DataFrame(columns=STOP_COLUMNS)
        ev = ev.sort_values(["message_ts", "offset"])
        sched = self.schedules.get(rid, {})
        ssd = pd.Timestamp(ev["ssd"].iloc[0]).date()
        rows: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for (tpl, etype), g in ev.groupby(["tiploc", "event_type"], sort=False):
            actual = g[g["time_kind"] == "actual"]
            last = actual.iloc[-1] if len(actual) else g.iloc[-1]
            planned = g["planned_time"].dropna()
            row = rows.setdefault(tpl, {})
            if tpl not in order:
                order.append(tpl)
            row[f"sched_{etype}"] = planned.iloc[-1] if len(planned) else pd.NaT
            row[f"actual_{etype}"] = last["time"] if last["time_kind"] == "actual" else pd.NaT
            reasons = g["late_reason"].dropna()
            if len(reasons):
                row["late_canc_reason"] = reasons.iloc[-1]
        cancelled, _ = self._cancelled(sched)
        toc = sched.get("toc")
        toc = None if toc is None or pd.isna(toc) else str(toc)
        table = []
        for tpl in order:
            r = rows[tpl]
            first = min(
                (t for t in (r.get("sched_arr"), r.get("sched_dep")) if pd.notna(t)),
                default=pd.NaT,
            )
            table.append(
                {
                    "rid": rid,
                    "date_of_service": ssd,
                    "toc_code": toc,
                    "location": self.tiplocs.get(tpl, tpl),
                    "sched_arr": r.get("sched_arr", pd.NaT),
                    "sched_dep": r.get("sched_dep", pd.NaT),
                    "actual_arr": r.get("actual_arr", pd.NaT),
                    "actual_dep": r.get("actual_dep", pd.NaT),
                    "late_canc_reason": r.get("late_canc_reason"),
                    "candidate_cancelled": cancelled,
                    "_first": first,
                }
            )
        df = pd.DataFrame(table).sort_values("_first", kind="stable").reset_index(drop=True)
        df["seq"] = range(len(df))
        return df[STOP_COLUMNS]

    @staticmethod
    def _cancelled(sched: dict[str, Any]) -> tuple[bool, str | None]:
        """(cancelled, reason code) from a schedule row. Missing values arrive as NaN."""
        if not sched:
            return False, None
        reason = sched.get("cancel_reason")
        has_reason = reason is not None and not pd.isna(reason)
        n_stops, n_can = sched.get("n_stops"), sched.get("n_cancelled_stops")
        whole = n_stops is not None and not pd.isna(n_stops) and n_stops == n_can
        return bool(has_reason or whole), (str(reason) if has_reason else None)

    def cancellation(self, rid: str) -> tuple[bool, str | None]:
        return self._cancelled(self.schedules.get(rid, {}))


def resolve_route(stops: pd.DataFrame, routes: list[Any], prefer: str | None = None) -> str | None:
    """The first route whose origin is called at before its destination, or `prefer`."""
    locations = stops.sort_values("seq")["location"].tolist()
    matches = []
    for route in routes:
        if route.origin in locations and route.destination in locations:
            if locations.index(route.origin) < locations.index(route.destination):
                matches.append(route.id)
    if prefer is not None:
        return prefer if prefer in matches else None
    return matches[0] if matches else None


# serving features: the training code on a stop table


def _spark_stop_schema():
    from pyspark.sql import types as T

    ts = T.TimestampType()
    return T.StructType(
        [
            T.StructField("rid", T.StringType()),
            T.StructField("service_date", T.DateType()),
            T.StructField("toc_code", T.StringType()),
            T.StructField("seq", T.LongType()),
            T.StructField("location", T.StringType()),
            T.StructField("sched_arr", ts),
            T.StructField("sched_dep", ts),
            T.StructField("actual_arr", ts),
            T.StructField("actual_dep", ts),
            T.StructField("late_canc_reason", T.StringType()),
            T.StructField("candidate_cancelled", T.BooleanType()),
        ]
    )


_SPARK_STOP_SCHEMA = None


def _stop_tuples(stops: pd.DataFrame) -> list[tuple]:
    """Rows as plain Python values in the stop schema: no NaT, no pd.NA."""

    def py(v: Any) -> Any:
        if v is None or v is pd.NaT or (not isinstance(v, (str, bool)) and pd.isna(v)):
            return None
        return v

    out = []
    for r in stops.itertuples(index=False):
        day = pd.Timestamp(r.date_of_service).date()
        times = [None if pd.isna(t) else pd.Timestamp(t).to_pydatetime() for t in (
            r.sched_arr, r.sched_dep, r.actual_arr, r.actual_dep)]  # fmt: skip
        out.append(
            (
                str(r.rid),
                day,
                py(r.toc_code),
                int(r.seq),
                str(r.location),
                *times,
                py(r.late_canc_reason),
                bool(r.candidate_cancelled),
            )  # fmt: skip
        )
    return out


class FeatureService:
    """Builds snapshot features with the training `build_snapshots` function.

    A batch is one Spark job for any number of services, so the fixed cost of planning the
    query is paid once per batch rather than once per service.
    """

    def __init__(self, spark: Any, interim: Path, routes_file: Path) -> None:
        from rail_risk.features.snapshots import (
            destination_services,
            history_labels,
            load_rid_routes,
            load_stops,
            route_destinations,
        )

        route_dest = route_destinations(routes_file)
        stops = load_stops(spark, interim / "_unified" / "hsp_stops.parquet")
        rid_routes = load_rid_routes(spark, interim / "hsp_manifest.sqlite")
        services, _ = destination_services(spark, stops, rid_routes, route_dest)
        self._init(spark, route_dest, history_labels(services))

    def _init(self, spark: Any, route_dest: dict[str, str], labels: Any) -> None:
        self.spark = spark
        self.route_dest = route_dest
        self.labels = labels.cache()
        self.labels.count()  # materialise the history once

    @classmethod
    def from_labels(cls, spark: Any, route_dest: dict[str, str], labels: Any) -> "FeatureService":
        """For tests and tools that already hold the history labels as a Spark frame."""
        obj = cls.__new__(cls)
        obj._init(spark, route_dest, labels)
        return obj

    def batch(
        self, services: dict[str, tuple[pd.DataFrame, str, int]]
    ) -> dict[str, dict[str, Any]]:
        """Features for {rid: (stop table, route id, minutes before scheduled arrival)}."""
        from rail_risk.features.snapshots import build_snapshots

        if not services:
            return {}
        tables, pairs = [], []
        for rid, (table, route_id, _minutes) in services.items():
            t = table.copy()
            for col in ("sched_arr", "sched_dep", "actual_arr", "actual_dep"):
                t[col] = pd.to_datetime(t[col], utc=True)
            tables.append(t)
            pairs.append((rid, route_id))
        stops = pd.concat(tables, ignore_index=True)
        global _SPARK_STOP_SCHEMA
        if _SPARK_STOP_SCHEMA is None:
            _SPARK_STOP_SCHEMA = _spark_stop_schema()
        sdf = self.spark.createDataFrame(_stop_tuples(stops), _SPARK_STOP_SCHEMA)
        rid_routes = self.spark.createDataFrame(pairs, "rid string, route_id string")
        horizons = tuple(sorted({int(m) for _, _, m in services.values()}))
        snap = build_snapshots(
            self.spark, sdf, rid_routes, self.route_dest, horizons=horizons, labels=self.labels
        )
        want = {rid: int(m) for rid, (_, _, m) in services.items()}
        out: dict[str, dict[str, Any]] = {}
        for row in snap.collect():
            d = row.asDict()
            if want.get(d["rid"]) == d["horizon_min"]:
                out[d["rid"]] = d
        return out

    def features(self, stop_table: pd.DataFrame, route_id: str, minutes: int) -> dict[str, Any]:
        rid = str(stop_table["rid"].iloc[0])
        return self.batch({rid: (stop_table, route_id, minutes)}).get(rid, {})


# data checks between the two feeds

LATENESS_FEATURES = ("last_stop_lateness_min", "lateness_change_last_2")
EXTRA_PARITY_FEATURES = ("n_stops_passed", "stops_remaining", "minutes_since_last_stop")


def _close(a: Any, b: Any, tol: float) -> bool:
    if (a is None or pd.isna(a)) and (b is None or pd.isna(b)):
        return True
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= tol


def source_parity(
    features: FeatureService,
    pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    route_id: str,
    minutes: int = 60,
    tol: float = 1.0,
) -> dict[str, Any]:
    """Features for the same services from HSP and from Darwin, through the same function.

    `pairs` maps rid to (HSP stop table, Darwin stop table). Both are given to `build_snapshots`
    at `minutes` before the scheduled arrival. A service agrees when every lateness feature is
    within `tol` minutes (or null in both). Returns the share and the per-feature shares.
    """
    hsp = features.batch({rid: (a, route_id, minutes) for rid, (a, _) in pairs.items()})
    darwin = features.batch({rid: (b, route_id, minutes) for rid, (_, b) in pairs.items()})
    rids = sorted(set(hsp) & set(darwin))
    per: dict[str, list[bool]] = {c: [] for c in (*LATENESS_FEATURES, *EXTRA_PARITY_FEATURES)}
    agree = []
    detail = []
    for rid in rids:
        flags = {c: _close(hsp[rid].get(c), darwin[rid].get(c), tol) for c in per}
        for c, ok in flags.items():
            per[c].append(ok)
        ok_all = all(flags[c] for c in LATENESS_FEATURES)
        agree.append(ok_all)
        if not ok_all:
            detail.append(
                {
                    "rid": rid,
                    **{f"hsp_{c}": hsp[rid].get(c) for c in LATENESS_FEATURES},
                    **{f"darwin_{c}": darwin[rid].get(c) for c in LATENESS_FEATURES},
                }
            )
    n = len(rids)
    return {
        "services": n,
        "agree_share": (sum(agree) / n) if n else None,
        "per_feature_share": {c: (sum(v) / n if n else None) for c, v in per.items()},
        "disagreements": detail,
    }


def actual_time_agreement(
    pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]], tol_seconds: float = 60.0
) -> dict[str, Any]:
    """How often the two feeds give the same actual time for the same planned stop event.

    Stops are matched on the planned (scheduled) time of the arrival or departure, because
    Darwin uses TIPLOCs and HSP uses CRS codes.
    """
    compared = exact = within = only_hsp = only_darwin = 0
    for _, (h, d) in pairs.items():
        for kind in ("arr", "dep"):
            sched, actual = f"sched_{kind}", f"actual_{kind}"
            hh = h[h[sched].notna()].set_index(sched)[actual]
            dd = d[d[sched].notna()].set_index(sched)[actual]
            for key, a in hh.items():
                if key not in dd.index:
                    continue
                b = dd.loc[key]
                b = b.iloc[-1] if isinstance(b, pd.Series) else b
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a):
                    only_darwin += 1
                    continue
                if pd.isna(b):
                    only_hsp += 1
                    continue
                compared += 1
                gap = abs((pd.Timestamp(a) - pd.Timestamp(b)).total_seconds())
                exact += gap == 0
                within += gap <= tol_seconds
    return {
        "compared_actuals": compared,
        "identical_share": (exact / compared) if compared else None,
        "within_tolerance_share": (within / compared) if compared else None,
        "tolerance_seconds": tol_seconds,
        "actual_in_hsp_only": only_hsp,
        "actual_in_darwin_only": only_darwin,
    }
