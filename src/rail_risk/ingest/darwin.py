"""Parse captured Darwin Push Port files into validated Parquet tables.

Reads data/raw/darwin/YYYY-MM-DD/HH.jsonl.gz tolerantly and writes three tables under
data/interim/darwin/, partitioned by date: darwin_schedules, darwin_events and
darwin_associations, each with a quarantine table of rows that failed parsing or validation.
Nothing is filtered to our routes here.
"""

import json
import re
import sqlite3
import time
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
import pyarrow as arrow
import pyarrow.parquet as pq

from rail_risk.ingest.times import (
    LONDON,
    ROLLOVER_MIN_DROP,
    nearest_day,
    parse_clock,
    to_utc,
    to_utc_near,
)
from rail_risk.logging import get_logger

TABLES = ("darwin_schedules", "darwin_events", "darwin_associations")
CHUNK_ROWS = 200_000
META_KEYS = {"updateOrigin", "requestSource", "requestID"}
STOP_KINDS = ("OR", "OPOR", "IP", "OPIP", "PP", "OPDT", "DT")
_FRACTION = re.compile(r"(\.\d{6})\d+")


# reading


@dataclass
class ReadStats:
    lines: int = 0
    duplicates: int = 0
    unreadable: int = 0
    bytes_lost: int = 0
    truncated: bool = False
    corrupt: bool = False


def iter_lines(path: Path, stats: ReadStats) -> Iterator[bytes]:
    """Complete lines from a possibly truncated, multi-member gzip file.

    A partial final line, or data after a corrupt point, is dropped and counted in bytes_lost.
    """
    data = path.read_bytes()
    decomp = zlib.decompressobj(31)
    in_member = False  # the current member has input but has not reached its end
    buf = b""
    pos = 0
    step = 1 << 16
    try:
        while pos < len(data):
            chunk = data[pos : pos + step]
            pos += len(chunk)
            while chunk:
                buf += decomp.decompress(chunk)
                in_member = True
                *lines, buf = buf.split(b"\n")
                yield from lines
                if decomp.eof:
                    chunk = decomp.unused_data
                    decomp = zlib.decompressobj(31)
                    in_member = False
                else:
                    chunk = b""
    except zlib.error:
        stats.corrupt = True
        stats.bytes_lost += len(buf) + (len(data) - pos)
        return
    if in_member:
        stats.truncated = True
    if buf:
        stats.truncated = True
        stats.bytes_lost += len(buf)


def normalise(line: bytes) -> tuple[int, int, int | None, dict[str, Any] | None, str | None]:
    """(partition, offset, kafka_ts_ms, payload, problem) for either stored line format."""
    rec = json.loads(line)
    partition, offset = int(rec["partition"]), int(rec["offset"])
    if "payload" in rec:
        return partition, offset, rec.get("kafka_ts"), rec["payload"], None
    if "raw" in rec:
        return partition, offset, rec.get("kafka_ts"), None, "unparsed payload"
    wrapper = rec["value"]  # first trial format: the JMS wrapper
    payload = json.loads(wrapper["bytes"])
    return partition, offset, None, payload, None


# field helpers


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def reason(value: Any) -> tuple[str | None, str | None]:
    """(code, tiploc) from a reason that is a bare code or an object with the code under ''."""
    if value is None:
        return None, None
    if isinstance(value, dict):
        code = value.get("")
        return (str(code) if code is not None else None), value.get("tiploc")
    return str(value), None


def message_time(payload: dict[str, Any], kafka_ts: int | None) -> datetime:
    text = payload.get("ts")
    if text:
        return datetime.fromisoformat(_FRACTION.sub(r"\1", text)).astimezone(UTC)
    if kafka_ts is not None:
        return datetime.fromtimestamp(kafka_ts / 1000, tz=UTC)
    raise ValueError("message has no timestamp")


def _is_true(value: Any) -> bool:
    return str(value).lower() == "true"


class Chain:
    """Resolves HH:MM[:SS] clock times along a running order, across midnight and DST."""

    def __init__(self, ssd: date) -> None:
        self.prev = datetime.combine(ssd, datetime.min.time())
        self.prev_utc: datetime | None = None

    def resolve(self, text: str | None) -> tuple[datetime, datetime] | None:
        t = parse_clock(text)
        if t is None:
            return None
        dt = datetime.combine(self.prev.date(), t)
        if self.prev - dt > ROLLOVER_MIN_DROP:
            dt += timedelta(days=1)
        utc = to_utc(dt, self.prev_utc)
        self.prev, self.prev_utc = dt, utc
        return dt, utc


# message parsing


@dataclass
class Parsed:
    schedules: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    associations: list[dict[str, Any]] = field(default_factory=list)
    quarantine: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {t: [] for t in (*TABLES, "darwin_messages")}
    )


def parse_schedule(el: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    ssd = date.fromisoformat(el["ssd"])
    raw_stops: list[dict[str, Any]] = []
    for kind in STOP_KINDS:
        for s in as_list(el.get(kind)):
            raw_stops.append({"kind": kind, **s})
    if not raw_stops:
        raise ValueError("schedule has no stops")

    def ref(s: dict[str, Any]) -> str | None:
        return next((s[k] for k in ("wta", "wtp", "wtd", "pta", "ptd") if s.get(k)), None)

    # The JSON groups stops by tag (all IP, then all PP, then DT), which loses the running order,
    # and a divided train has several DT entries. Restore an order by time since the origin.
    origin = [s for s in raw_stops if s["kind"] in ("OR", "OPOR")][:1]
    rest = [s for s in raw_stops if s not in origin]
    first = (
        parse_clock(next((origin[0][k] for k in ("wtd", "ptd") if origin[0].get(k)), None))
        if origin
        else None
    )
    if first is not None:
        base = first.hour * 3600 + first.minute * 60 + first.second

        def offset(s: dict[str, Any]) -> int:
            t = parse_clock(ref(s))
            if t is None:
                return 0
            return (t.hour * 3600 + t.minute * 60 + t.second - base) % 86400

        rest.sort(key=offset)
    ordered = origin + rest

    chain = Chain(ssd)
    stops = []
    for seq, s in enumerate(ordered):
        times: dict[str, datetime | None] = {}
        for key in ("wta", "pta", "wtp", "wtd", "ptd"):
            got = chain.resolve(s.get(key))
            times[key] = got[1] if got else None
        stops.append(
            {
                "seq": seq,
                "tpl": s["tpl"],
                "kind": s["kind"],
                "act": s.get("act"),
                "can": _is_true(s.get("can")),
                **times,
            }
        )
    code, code_tpl = reason(el.get("cancelReason"))
    return {
        "rid": el["rid"],
        "uid": el.get("uid"),
        "ssd": pd.Timestamp(ssd),
        "train_id": el.get("trainId"),
        "toc": el.get("toc"),
        "origin_tpl": stops[0]["tpl"],
        "dest_tpl": stops[-1]["tpl"],  # a divided train has several DT; this is the latest
        "cancel_reason": code,
        "cancel_reason_tpl": code_tpl,
        "n_stops": len(stops),
        "n_cancelled_stops": sum(1 for s in stops if s["can"]),
        "stops": stops,
        **ctx,
    }


def parse_ts(el: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    ssd = date.fromisoformat(el["ssd"])
    late_code, late_tpl = reason(el.get("LateReason"))
    anchor_utc: datetime = ctx["message_ts"]
    anchor_local = anchor_utc.astimezone(LONDON).replace(tzinfo=None)
    chain = Chain(ssd)
    rows = []
    for loc in as_list(el.get("Location")):
        planned_field = {"arr": ("pta", "wta"), "dep": ("ptd", "wtd"), "pass": ("wtp",)}
        for event_type in ("arr", "dep", "pass"):
            ev = loc.get(event_type)
            if ev is None:
                continue
            planned = None
            for key in planned_field[event_type]:
                planned = chain.resolve(loc.get(key))
                if planned:
                    break
            if isinstance(ev, str):
                ev = {"et": ev}
            kind, text = "unknown", None
            if ev.get("at"):
                kind, text = "actual", ev["at"]
            elif ev.get("et"):
                kind, text = "forecast", ev["et"]
            when = None
            clock = parse_clock(text)
            if clock is not None:
                local = nearest_day(clock, planned[0] if planned else anchor_local)
                when = to_utc_near(local, planned[1] if planned else anchor_utc)
            rows.append(
                {
                    "rid": el["rid"],
                    "uid": el.get("uid"),
                    "ssd": pd.Timestamp(ssd),
                    "tiploc": loc["tpl"],
                    "event_type": event_type,
                    "time_kind": kind,
                    "time": when,
                    "planned_time": planned[1] if planned else None,
                    "src": ev.get("src"),
                    "delayed": _is_true(ev.get("delayed")),
                    "late_reason": late_code,
                    "late_reason_tpl": late_tpl,
                    **ctx,
                }
            )
    return rows


def parse_association(el: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": el.get("category"),
        "tiploc": el["tiploc"],
        "main_rid": el["main"]["rid"],
        "assoc_rid": el["assoc"]["rid"],
        "is_cancelled": _is_true(el.get("isCancelled")),
        "is_deleted": _is_true(el.get("isDeleted")),
        **ctx,
    }


def parse_payload(
    payload: dict[str, Any], partition: int, offset: int, kafka_ts: int | None, out: Parsed
) -> None:
    ur = payload.get("uR")
    if not isinstance(ur, dict):
        return  # heartbeat responses and other non-update messages
    try:
        ctx = {
            "message_ts": message_time(payload, kafka_ts),
            "partition": partition,
            "offset": offset,
        }
    except ValueError as exc:
        out.quarantine["darwin_messages"].append(_q(partition, offset, str(exc), payload))
        return
    for el in as_list(ur.get("schedule")):
        try:
            row = parse_schedule(el, ctx)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            out.quarantine["darwin_schedules"].append(_q(partition, offset, repr(exc), el))
            continue
        old = out.schedules.get(row["rid"])
        if old is None or (row["message_ts"], offset) >= (old["message_ts"], old["offset"]):
            out.schedules[row["rid"]] = row
    for el in as_list(ur.get("TS")):
        try:
            out.events.extend(parse_ts(el, ctx))
        except (ValueError, KeyError, TypeError) as exc:
            out.quarantine["darwin_events"].append(_q(partition, offset, repr(exc), el))
    for el in as_list(ur.get("association")):
        try:
            out.associations.append(parse_association(el, ctx))
        except (ValueError, KeyError, TypeError) as exc:
            out.quarantine["darwin_associations"].append(_q(partition, offset, repr(exc), el))


def _q(partition: int, offset: int, why: str, source: Any) -> dict[str, Any]:
    return {
        "partition": partition,
        "offset": offset,
        "quarantine_reason": why,
        "raw": json.dumps(source, separators=(",", ":"))[:20000],
    }


# validation

_RID = r"^\d{15}$"
_TIPLOC = r"^[A-Z0-9]{1,7}$"
_UTC = "datetime64[ns, UTC]"

SCHEDULES_SCHEMA = pa.DataFrameSchema(
    {
        "rid": pa.Column(str, pa.Check.str_matches(_RID)),
        "uid": pa.Column(str, nullable=True),
        "ssd": pa.Column(pa.DateTime),
        "train_id": pa.Column(str, nullable=True),
        "toc": pa.Column(str, pa.Check.str_matches(r"^[A-Z0-9]{2}$"), nullable=True),
        "origin_tpl": pa.Column(str, pa.Check.str_matches(_TIPLOC)),
        "dest_tpl": pa.Column(str, pa.Check.str_matches(_TIPLOC)),
        "cancel_reason": pa.Column(str, pa.Check.str_matches(r"^\d+$"), nullable=True),
        "cancel_reason_tpl": pa.Column(str, nullable=True),
        "n_stops": pa.Column(int, pa.Check.ge(1)),
        "n_cancelled_stops": pa.Column(int, pa.Check.ge(0)),
        "stops": pa.Column(object),
        "message_ts": pa.Column(_UTC),
        "partition": pa.Column(int, pa.Check.ge(0)),
        "offset": pa.Column(int, pa.Check.ge(0)),
    },
    unique=["rid"],
    strict=True,
)

EVENTS_SCHEMA = pa.DataFrameSchema(
    {
        "rid": pa.Column(str, pa.Check.str_matches(_RID)),
        "uid": pa.Column(str, nullable=True),
        "ssd": pa.Column(pa.DateTime),
        "tiploc": pa.Column(str, pa.Check.str_matches(_TIPLOC)),
        "event_type": pa.Column(str, pa.Check.isin(["arr", "dep", "pass"])),
        "time_kind": pa.Column(str, pa.Check.isin(["forecast", "actual", "unknown"])),
        "time": pa.Column(_UTC, nullable=True),
        "planned_time": pa.Column(_UTC, nullable=True),
        "src": pa.Column(str, nullable=True),
        "delayed": pa.Column(bool),
        "late_reason": pa.Column(str, pa.Check.str_matches(r"^\d+$"), nullable=True),
        "late_reason_tpl": pa.Column(str, nullable=True),
        "message_ts": pa.Column(_UTC),
        "partition": pa.Column(int, pa.Check.ge(0)),
        "offset": pa.Column(int, pa.Check.ge(0)),
    },
    checks=[
        pa.Check(
            lambda df: (
                df["time"].isna()
                | (
                    (df["time"] >= df["ssd"].dt.tz_localize("UTC") - pd.Timedelta(days=1))
                    & (df["time"] <= df["ssd"].dt.tz_localize("UTC") + pd.Timedelta(days=3))
                )
            ),
            name="time_within_service_window",
        )
    ],
    strict=True,
)

ASSOCIATIONS_SCHEMA = pa.DataFrameSchema(
    {
        "category": pa.Column(str, pa.Check.isin(["JJ", "VV", "LK", "NP"])),
        "tiploc": pa.Column(str, pa.Check.str_matches(_TIPLOC)),
        "main_rid": pa.Column(str, pa.Check.str_matches(_RID)),
        "assoc_rid": pa.Column(str, pa.Check.str_matches(_RID)),
        "is_cancelled": pa.Column(bool),
        "is_deleted": pa.Column(bool),
        "message_ts": pa.Column(_UTC),
        "partition": pa.Column(int, pa.Check.ge(0)),
        "offset": pa.Column(int, pa.Check.ge(0)),
    },
    strict=True,
)

SCHEMAS = {
    "darwin_schedules": SCHEDULES_SCHEMA,
    "darwin_events": EVENTS_SCHEMA,
    "darwin_associations": ASSOCIATIONS_SCHEMA,
}

STOP_STRUCT = arrow.struct(
    [
        ("seq", arrow.int32()),
        ("tpl", arrow.string()),
        ("kind", arrow.string()),
        ("act", arrow.string()),
        ("can", arrow.bool_()),
        *[(k, arrow.timestamp("us", tz="UTC")) for k in ("wta", "pta", "wtp", "wtd", "ptd")],
    ]
)


def to_frame(table: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("message_ts", "time", "planned_time"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True).astype(_UTC)
    if "ssd" in df:
        df["ssd"] = pd.to_datetime(df["ssd"]).astype("datetime64[ns]")
    for col in ("partition", "offset"):
        df[col] = df[col].astype("int64")
    if table == "darwin_schedules":
        for col in ("n_stops", "n_cancelled_stops"):
            df[col] = df[col].astype("int64")
    return df


def validate(table: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Valid rows, plus quarantine records for rows that fail the schema."""
    if df.empty:
        return df, []
    df = df.reset_index(drop=True)
    try:
        SCHEMAS[table].validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        why: dict[int, set[str]] = {}
        for _, f in exc.failure_cases.iterrows():
            label = (
                str(f["check"])
                if f["schema_context"] == "DataFrameSchema"
                else f"{f['column']}: {f['check']}"
            )
            targets = range(len(df)) if pd.isna(f["index"]) else [int(f["index"])]
            for i in targets:
                why.setdefault(i, set()).add(label)
        bad = sorted(why)
        quarantine = [
            {
                "partition": int(df.at[i, "partition"]),
                "offset": int(df.at[i, "offset"]),
                "quarantine_reason": "; ".join(sorted(why[i])),
                "raw": json.dumps(df.loc[i].to_dict(), default=str)[:20000],
            }
            for i in bad
        ]
        return df.drop(index=bad).reset_index(drop=True), quarantine
    return df, []


# writing


def _write(df: pd.DataFrame, table: str, partition_col: str | None, root: Path, key: str) -> None:
    if df.empty:
        return
    if partition_col is None:
        groups = [("unknown", df)]
    else:
        days = df[partition_col].dt.strftime("%Y-%m-%d")
        groups = list(df.groupby(days))
    for day, part in groups:
        path = root / table / f"date={day}" / f"{key}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if table == "darwin_schedules":
            tbl = arrow.Table.from_pandas(part, preserve_index=False)
            idx = tbl.schema.get_field_index("stops")
            stops_col = arrow.array(
                [list(map(dict, s)) for s in part["stops"]], type=arrow.list_(STOP_STRUCT)
            )
            tbl = tbl.set_column(idx, "stops", stops_col)
            pq.write_table(tbl, path)
        else:
            part.to_parquet(path, index=False)


def _remove_old_parts(root: Path, key: str) -> None:
    for pattern in (f"*/date=*/{key}_*.parquet", f"*_quarantine/{key}_*.parquet"):
        for path in root.glob(pattern):
            path.unlink()


# incremental driver


@dataclass
class FileResult:
    key: str
    seconds: float
    stats: ReadStats
    rows: dict[str, int]
    quarantined: dict[str, int]
    reasons: dict[str, int]


class DarwinManifest:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS files (
                key TEXT PRIMARY KEY, size INTEGER, parsed_at TEXT, seconds REAL, lines INTEGER,
                duplicates INTEGER, unreadable INTEGER, bytes_lost INTEGER,
                schedules INTEGER, events INTEGER, associations INTEGER,
                q_schedules INTEGER, q_events INTEGER, q_associations INTEGER,
                q_messages INTEGER)"""
        )

    def size_of(self, key: str) -> int | None:
        row = self._db.execute("SELECT size FROM files WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def record(self, key: str, size: int, r: FileResult) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,datetime('now'),?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key, size, r.seconds, r.stats.lines, r.stats.duplicates, r.stats.unreadable,
                    r.stats.bytes_lost, r.rows["darwin_schedules"], r.rows["darwin_events"],
                    r.rows["darwin_associations"], r.quarantined["darwin_schedules"],
                    r.quarantined["darwin_events"], r.quarantined["darwin_associations"],
                    r.quarantined["darwin_messages"],
                ),
            )  # fmt: skip

    def totals(self) -> tuple[Any, ...]:
        return self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(schedules),0), COALESCE(SUM(events),0), "
            "COALESCE(SUM(associations),0), "
            "COALESCE(SUM(q_schedules+q_events+q_associations+q_messages),0), "
            "COALESCE(SUM(duplicates),0), COALESCE(SUM(bytes_lost),0), MAX(parsed_at) FROM files"
        ).fetchone()

    def close(self) -> None:
        self._db.close()


def parse_file(path: Path, key: str, interim_root: Path) -> FileResult:
    started = time.monotonic()
    root = interim_root / "darwin"
    _remove_old_parts(root, key)
    stats = ReadStats()
    out = Parsed()
    seen: set[tuple[int, int]] = set()
    rows = dict.fromkeys((*TABLES, "darwin_messages"), 0)
    quarantined = dict.fromkeys((*TABLES, "darwin_messages"), 0)
    reasons: dict[str, int] = {}
    chunk = 0

    def flush(final: bool) -> None:
        nonlocal chunk
        batches = {"darwin_events": out.events, "darwin_associations": out.associations}
        if final:
            batches["darwin_schedules"] = list(out.schedules.values())
        for table, items in batches.items():
            good, bad = validate(table, to_frame(table, items))
            rows[table] += len(good)
            out.quarantine[table].extend(bad)
            _write(
                good, table, "ssd" if table != "darwin_associations" else "message_ts", root,
                f"{key}_{chunk}",
            )  # fmt: skip
            items.clear()
        chunk += 1
        for table, items in out.quarantine.items():
            if items:
                quarantined[table] += len(items)
                for item in items:
                    reasons[item["quarantine_reason"]] = (
                        reasons.get(item["quarantine_reason"], 0) + 1
                    )
                df = pd.DataFrame(items)
                path_ = root / f"{table}_quarantine" / f"{key}_{chunk}.parquet"
                path_.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path_, index=False)
                items.clear()

    for line in iter_lines(path, stats):
        if not line.strip():
            continue
        stats.lines += 1
        try:
            partition, offset, kafka_ts, payload, problem = normalise(line)
        except (ValueError, KeyError, TypeError) as exc:
            stats.unreadable += 1
            out.quarantine["darwin_messages"].append(_q(-1, -1, f"unreadable line: {exc!r}", ""))
            continue
        if (partition, offset) in seen:
            stats.duplicates += 1
            continue
        seen.add((partition, offset))
        if payload is None:
            out.quarantine["darwin_messages"].append(_q(partition, offset, str(problem), ""))
            continue
        parse_payload(payload, partition, offset, kafka_ts, out)
        if len(out.events) >= CHUNK_ROWS:
            flush(final=False)
    flush(final=True)
    return FileResult(key, time.monotonic() - started, stats, rows, quarantined, reasons)


def parse_directory(
    raw_root: Path,
    interim_root: Path,
    manifest: DarwinManifest,
    *,
    only_date: date | None = None,
    now: datetime | None = None,
) -> list[FileResult]:
    """Parse every closed hourly file once. A file that has grown since is parsed again."""
    log = get_logger("darwin")
    current = (now or datetime.now(UTC)).strftime("%Y-%m-%d/%H")
    results = []
    for path in sorted(raw_root.glob("*/*.jsonl.gz")):
        key = f"{path.parent.name}/{path.name.removesuffix('.jsonl.gz')}"
        if only_date and path.parent.name != only_date.isoformat():
            continue
        if key == current:
            log.info("skip_current_hour", file=key)
            continue
        size = path.stat().st_size
        if manifest.size_of(key) == size:
            continue
        result = parse_file(path, key.replace("/", "T"), interim_root)
        manifest.record(key, size, result)
        results.append(result)
        log.info(
            "file_parsed",
            file=key,
            seconds=round(result.seconds, 1),
            lines=result.stats.lines,
            duplicates=result.stats.duplicates,
            bytes_lost=result.stats.bytes_lost,
            rows=result.rows,
            quarantined=result.quarantined,
        )
        if result.stats.bytes_lost:
            log.warning("bytes_lost", file=key, bytes=result.stats.bytes_lost)
    return results
