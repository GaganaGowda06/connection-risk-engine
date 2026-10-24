"""Streaming parser for the community Darwin Push Port archive (XML v18, one gzip per hour).

An hour file is a concatenation of XML documents, one Pport message each, so it is read as a
stream of documents and each one is parsed with lxml; nothing is ever decompressed to disk.
One call handles one file and writes Parquet parts per service date, so a file can be redone
alone. Times are UTC timestamps. Darwin clock times are UK local times on the service date, so a
time is placed on the day nearest the message time (updates) or by a rolling day count along the
schedule, and both clock-change nights are handled with the UK offset in force.
"""

import gzip
import hashlib
import json
import re
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from lxml import etree

from rail_risk.logging import get_logger

PARSER_VERSION = "1"
LONDON = ZoneInfo("Europe/London")
EPOCH_ORD = date(1970, 1, 1).toordinal()
DAY = 86400
BATCH_ROWS = 200_000
TS_UTC = pa.timestamp("us", tz="UTC")
NS = re.compile(r"^\{[^}]*\}")


def local_name(tag: Any) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


# ---------------------------------------------------------------- time handling


@cache
def _day_offset(ordinal: int) -> int | None:
    """UK UTC offset in seconds for the whole local day, or None on a clock-change day."""
    d = date.fromordinal(ordinal)
    start = datetime(d.year, d.month, d.day, 0, 0, tzinfo=LONDON).utcoffset()
    end = datetime(d.year, d.month, d.day, 23, 59, tzinfo=LONDON).utcoffset()
    return int(start.total_seconds()) if start == end else None


def local_to_utc(ordinal: int, secs: int) -> list[int]:
    """UTC epoch seconds for a local clock time (secs from midnight, may be 24h or more).

    One value on an ordinary day, two for a repeated time on the autumn change day, and for a
    time that does not exist (spring change day) the single value zoneinfo gives.
    """
    off = _day_offset(ordinal)
    base = (ordinal - EPOCH_ORD) * DAY
    if off is not None:
        return [base + secs - off]
    d = date.fromordinal(ordinal)
    naive = datetime(d.year, d.month, d.day) + timedelta(seconds=secs)
    return sorted({int(naive.replace(tzinfo=LONDON, fold=f).timestamp()) for f in (0, 1)})


@lru_cache(maxsize=4096)
def clock_secs(text: str) -> int:
    parts = text.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + (int(parts[2]) if len(parts) > 2 else 0)


def parse_message_time(text: str) -> tuple[int, int, int, int]:
    """(utc_epoch_us, local ordinal, local seconds of day, utc offset seconds) of a Pport ts."""
    y, mo, d = int(text[0:4]), int(text[5:7]), int(text[8:10])
    h, mi, s = int(text[11:13]), int(text[14:16]), int(text[17:19])
    frac = 0
    rest = text[19:]
    if rest.startswith("."):
        digits = re.match(r"\.(\d+)", rest).group(1)
        frac = int((digits + "000000")[:6])
        rest = rest[1 + len(digits) :]
    if rest in ("Z", ""):
        off = 0
    else:
        off = (1 if rest[0] == "+" else -1) * (int(rest[1:3]) * 3600 + int(rest[4:6]) * 60)
    ordinal = date(y, mo, d).toordinal()
    secs = h * 3600 + mi * 60 + s
    utc = (ordinal - EPOCH_ORD) * DAY + secs - off
    return utc * 1_000_000 + frac, ordinal, secs, off


def nearest_utc(ordinal: int, secs: int, ref_utc: int) -> int:
    """UTC seconds of the local clock time, on the day (-1, 0 or +1) nearest the reference."""
    best = None
    for k in (0, -1, 1):
        for cand in local_to_utc(ordinal + k, secs):
            if best is None or abs(cand - ref_utc) < abs(best - ref_utc):
                best = cand
        if best is not None and abs(best - ref_utc) <= 12 * 3600 and k == 0:
            break
    return best


def schedule_times(ssd_ordinal: int, clocks: list[str | None]) -> list[int | None]:
    """Place the ordered clock times of one schedule on UTC, rolling the day when time wraps.

    A step back of more than six hours is a new day. A smaller step back is a repeated clock
    hour on the autumn change night, so the later of the two readings is used.
    """
    out: list[int | None] = []
    day = ssd_ordinal
    prev: int | None = None
    for text in clocks:
        if not text:
            out.append(None)
            continue
        secs = clock_secs(text)
        cands = local_to_utc(day, secs)
        cand = cands[0]
        if prev is not None:
            if cand < prev - 6 * 3600:
                day += 1
                cands = local_to_utc(day, secs)
                cand = cands[0]
            if len(cands) > 1 and cand < prev:
                cand = cands[1]
        out.append(cand)
        prev = cand
    return out


def service_day_times(ssd_ordinal: int, clocks: list[str | None], msg_utc: int) -> list[int | None]:
    """Place the ordered working-time clocks of one update on UTC, anchored to the service date.

    A planned time is on the service date or the day after. The first one is put on the day that
    is nearer the message time, with the message time held inside the service window so a
    forecast issued days ahead still lands on the service date; the rest roll along the order.
    """
    out: list[int | None] = []
    day = ssd_ordinal
    prev: int | None = None
    lo = local_to_utc(ssd_ordinal, 0)[0]
    hi = local_to_utc(ssd_ordinal + 2, 0)[0]
    ref = min(max(msg_utc, lo), hi)
    for text in clocks:
        if not text:
            out.append(None)
            continue
        secs = clock_secs(text)
        if prev is None:
            best = None
            for k in (0, 1):
                for cand in local_to_utc(ssd_ordinal + k, secs):
                    if best is None or abs(cand - ref) < abs(best[0] - ref):
                        best = (cand, ssd_ordinal + k)
            cand, day = best
        else:
            cands = local_to_utc(day, secs)
            cand = cands[0]
            if cand < prev - 6 * 3600:
                day += 1
                cands = local_to_utc(day, secs)
                cand = cands[0]
            if len(cands) > 1 and cand < prev:
                cand = cands[1]
        out.append(cand)
        prev = cand
    return out


def near_utc(secs: int, ref_utc: int) -> int:
    """UTC seconds of a local clock time on the day nearest a reference instant."""
    return nearest_utc((ref_utc + 3600) // DAY + EPOCH_ORD, secs, ref_utc)


def _us(utc_seconds: int | None) -> int | None:
    return None if utc_seconds is None else utc_seconds * 1_000_000


# ---------------------------------------------------------------- splits and config


@dataclass(frozen=True)
class Stage1Config:
    windows: list[tuple[date, date]]
    boundary_days: list[date]
    clock_change_days: list[date]
    splits: dict[str, tuple[date, date]]
    default_split: str


def load_stage1_config(path: Path = Path("config/stage1.yaml")) -> Stage1Config:
    raw = yaml.safe_load(path.read_text())
    return Stage1Config(
        windows=[(w["start"], w["end"]) for w in raw["windows"]],
        boundary_days=list(raw["boundary_days"]),
        clock_change_days=list(raw["clock_change_days"]),
        splits={k: (v["start"], v["end"]) for k, v in raw["splits"].items()},
        default_split=raw["default_split"],
    )


def split_for(ssd: date, cfg: Stage1Config) -> str:
    for name, (start, end) in cfg.splits.items():
        if start <= ssd <= end:
            return name
    return cfg.default_split


# ---------------------------------------------------------------- document stream


def iter_documents(fh) -> Any:
    """Yield each XML document of a Pport hour file (a message can span several lines)."""
    buf: list[bytes] = []
    for line in fh:
        if line.startswith(b"<?xml") and buf:
            yield b"".join(buf)
            buf = []
        buf.append(line)
    if buf:
        yield b"".join(buf)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- table schemas

_S = pa.string()
_B = pa.bool_()
SCHEMAS: dict[str, pa.Schema] = {
    "schedules": pa.schema(
        [
            ("rid", _S),
            ("uid", _S),
            ("ssd", pa.date32()),
            ("train_id", _S),
            ("toc", _S),
            ("status", _S),
            ("train_cat", _S),
            ("is_passenger", _B),
            ("is_active", _B),
            ("deleted", _B),
            ("is_charter", _B),
            ("cancel_reason", pa.int32()),
            ("cancel_tiploc", _S),
            ("cancel_near", _B),
            ("n_points", pa.int32()),
            ("update_origin", _S),
            ("message_time", TS_UTC),
            ("src_file", _S),
        ]
    ),
    "schedule_points": pa.schema(
        [
            ("rid", _S),
            ("ssd", pa.date32()),
            ("seq", pa.int32()),
            ("kind", _S),
            ("tpl", _S),
            ("act", _S),
            ("plan_act", _S),
            ("cancelled", _B),
            ("fd", _S),
            ("pt_arr", TS_UTC),
            ("pt_dep", TS_UTC),
            ("wt_arr", TS_UTC),
            ("wt_dep", TS_UTC),
            ("wt_pass", TS_UTC),
            ("rdelay", pa.int16()),
            ("message_time", TS_UTC),
            ("src_file", _S),
        ]
    ),
    "stop_updates": pa.schema(
        [
            ("rid", _S),
            ("ssd", pa.date32()),
            ("tpl", _S),
            ("event", _S),
            ("loc_seq", pa.int16()),
            ("pt_arr", TS_UTC),
            ("pt_dep", TS_UTC),
            ("wt_arr", TS_UTC),
            ("wt_dep", TS_UTC),
            ("wt_pass", TS_UTC),
            ("est", TS_UTC),
            ("est_min", TS_UTC),
            ("est_working", TS_UTC),
            ("actual", TS_UTC),
            ("actual_class", _S),
            ("src", _S),
            ("src_inst", _S),
            ("delayed", _B),
            ("est_unknown", _B),
            ("actual_removed", _B),
            ("plat", _S),
            ("plat_src", _S),
            ("plat_suppressed", _B),
            ("plat_confirmed", _B),
            ("length", pa.int16()),
            ("suppressed", _B),
            ("loc_late_reason", pa.int32()),
            ("ts_late_reason", pa.int32()),
            ("update_origin", _S),
            ("message_time", TS_UTC),
            ("src_file", _S),
        ]
    ),
    "associations": pa.schema(
        [
            ("main_rid", _S),
            ("assoc_rid", _S),
            ("ssd", pa.date32()),
            ("category", _S),
            ("tiploc", _S),
            ("is_cancelled", _B),
            ("is_deleted", _B),
            ("main_wta", TS_UTC),
            ("main_pta", TS_UTC),
            ("assoc_wtd", TS_UTC),
            ("assoc_ptd", TS_UTC),
            ("message_time", TS_UTC),
            ("src_file", _S),
        ]
    ),
    "deactivations": pa.schema(
        [("rid", _S), ("ssd", pa.date32()), ("message_time", TS_UTC), ("src_file", _S)]
    ),
}
for _name, _schema in list(SCHEMAS.items()):
    SCHEMAS[_name] = _schema.append(pa.field("split", _S))
SCHEMAS["stop_actuals"] = pa.schema(
    [
        ("rid", _S),
        ("ssd", pa.date32()),
        ("tpl", _S),
        ("event", _S),
        ("pt_arr", TS_UTC),
        ("pt_dep", TS_UTC),
        ("wt_arr", TS_UTC),
        ("wt_dep", TS_UTC),
        ("wt_pass", TS_UTC),
        ("actual", TS_UTC),
        ("actual_class", _S),
        ("src", _S),
        ("n_updates", pa.int32()),
        ("first_actual_message_time", TS_UTC),
        ("last_message_time", TS_UTC),
        ("split", _S),
    ]
)
TABLES = tuple(SCHEMAS)


def _bool(v: str | None) -> bool | None:
    return None if v is None else v.lower() == "true"


def _int(v: str | None) -> int | None:
    try:
        return None if v is None or v == "" else int(v)
    except ValueError:
        return None


def rid_ssd(rid: str) -> date | None:
    try:
        return date(int(rid[0:4]), int(rid[4:6]), int(rid[6:8]))
    except ValueError:
        return None


# ---------------------------------------------------------------- sinks


class Sink:
    """Buffers rows per (table, ssd) and writes one Parquet part per pair for this source file."""

    def __init__(self, out: Path, key: str, source_sha: str, source_name: str, cfg: "Stage1Config"):
        self.out, self.key, self.cfg = out, key, cfg
        self.meta = {
            b"source_sha256": source_sha.encode(),
            b"source_file": source_name.encode(),
            b"parser_version": PARSER_VERSION.encode(),
        }
        self.rows: dict[tuple[str, date], list[dict[str, Any]]] = {}
        self.writers: dict[tuple[str, date], pq.ParquetWriter] = {}
        self.counts: Counter = Counter()

    def add(self, table: str, ssd: date | None, row: dict[str, Any]) -> None:
        if ssd is None:
            self.counts[f"{table}_no_ssd"] += 1
            return
        bucket = self.rows.setdefault((table, ssd), [])
        bucket.append(row)
        self.counts[table] += 1
        if len(bucket) >= BATCH_ROWS:
            self._flush(table, ssd)

    def _path(self, table: str, ssd: date) -> Path:
        return self.out / table / f"service_date={ssd.isoformat()}" / f"{self.key}.parquet"

    def _flush(self, table: str, ssd: date) -> None:
        bucket = self.rows.get((table, ssd))
        if not bucket:
            return
        schema = SCHEMAS[table].with_metadata(self.meta)
        if (table, ssd) not in self.writers:
            path = self._path(table, ssd)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.writers[(table, ssd)] = pq.ParquetWriter(
                str(path) + ".tmp", schema, compression="zstd"
            )
        label = split_for(ssd, self.cfg)
        for r in bucket:
            r["split"] = label
        arrays = [pa.array([r.get(f.name) for r in bucket], type=f.type) for f in schema]
        self.writers[(table, ssd)].write_table(pa.Table.from_arrays(arrays, schema=schema))
        self.rows[(table, ssd)] = []

    def close(self, keep: bool = True) -> None:
        for table, ssd in list(self.rows):
            self._flush(table, ssd)
        for (table, ssd), writer in self.writers.items():
            writer.close()
            tmp = Path(str(self._path(table, ssd)) + ".tmp")
            if keep:
                tmp.rename(self._path(table, ssd))
            else:
                tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------- element handlers


def handle_schedule(el, msg, sink: Sink, src: str) -> None:
    rid = el.get("rid")
    ssd = _date(el.get("ssd")) or rid_ssd(rid or "")
    if not rid or ssd is None:
        sink.counts["schedule_bad"] += 1
        return
    points = [c for c in el if local_name(c.tag) in POINT_KINDS]
    clocks: list[str | None] = []
    for p in points:
        clocks += [p.get("wta"), p.get("pta"), p.get("wtp"), p.get("wtd"), p.get("ptd")]
    placed = schedule_times(ssd.toordinal(), clocks)
    cancel = next((c for c in el if local_name(c.tag) == "cancelReason"), None)
    msg_us = msg["us"]
    sink.add(
        "schedules",
        ssd,
        {
            "rid": rid,
            "uid": el.get("uid"),
            "ssd": ssd,
            "train_id": el.get("trainId"),
            "toc": el.get("toc"),
            "status": el.get("status"),
            "train_cat": el.get("trainCat"),
            "is_passenger": _default_true(el.get("isPassengerSvc")),
            "is_active": _default_true(el.get("isActive")),
            "deleted": _bool(el.get("deleted")) or False,
            "is_charter": _bool(el.get("isCharter")) or False,
            "cancel_reason": _int(cancel.text) if cancel is not None else None,
            "cancel_tiploc": cancel.get("tiploc") if cancel is not None else None,
            "cancel_near": _bool(cancel.get("near")) if cancel is not None else None,
            "n_points": len(points),
            "update_origin": msg["origin"],
            "message_time": msg_us,
            "src_file": src,
        },
    )
    for i, p in enumerate(points):
        wta, pta, wtp, wtd, ptd = (_us(t) for t in placed[i * 5 : i * 5 + 5])
        sink.add(
            "schedule_points",
            ssd,
            {
                "rid": rid,
                "ssd": ssd,
                "seq": i,
                "kind": local_name(p.tag),
                "tpl": p.get("tpl"),
                "act": p.get("act"),
                "plan_act": p.get("planAct"),
                "cancelled": _bool(p.get("can")) or False,
                "fd": p.get("fd"),
                "pt_arr": pta,
                "pt_dep": ptd,
                "wt_arr": wta,
                "wt_dep": wtd,
                "wt_pass": wtp,
                "rdelay": _int(p.get("rdelay")),
                "message_time": msg_us,
                "src_file": src,
            },
        )


POINT_KINDS = {"OR", "OPOR", "IP", "OPIP", "PP", "DT", "OPDT"}


def _default_true(v: str | None) -> bool:
    return True if v is None else v.lower() == "true"


def _date(text: str | None) -> date | None:
    try:
        return date.fromisoformat(text) if text else None
    except ValueError:
        return None


def handle_ts(el, msg, sink: Sink, src: str) -> None:
    rid = el.get("rid")
    ssd = _date(el.get("ssd")) or rid_ssd(rid or "")
    if not rid or ssd is None:
        sink.counts["ts_bad"] += 1
        return
    ref = msg["us"] // 1_000_000
    ts_late = None
    locs = []
    for child in el:
        name = local_name(child.tag)
        if name == "LateReason":
            ts_late = _int(child.text)
        elif name == "Location":
            locs.append(child)
    primary = service_day_times(
        ssd.toordinal(),
        [
            loc.get("wta") or loc.get("wtp") or loc.get("wtd") or loc.get("pta") or loc.get("ptd")
            for loc in locs
        ],
        ref,
    )
    for seq, loc in enumerate(locs):
        anchor = primary[seq] if primary[seq] is not None else ref
        events = []
        plat = plat_src = None
        plat_sup = plat_conf = None
        length = None
        suppressed = False
        loc_late = None
        for c in loc:
            n = local_name(c.tag)
            if n in ("arr", "dep", "pass"):
                events.append((n, c))
            elif n == "plat":
                plat, plat_src = (c.text or None), c.get("platsrc")
                plat_sup, plat_conf = _bool(c.get("platsup")), _bool(c.get("conf"))
            elif n == "length":
                length = _int(c.text)
            elif n == "suppr":
                suppressed = (c.text or "").strip().lower() == "true"
            elif n == "LateReason":
                loc_late = _int(c.text)
        planned = {}
        for col, key in (
            ("pt_arr", "pta"),
            ("pt_dep", "ptd"),
            ("wt_arr", "wta"),
            ("wt_dep", "wtd"),
            ("wt_pass", "wtp"),
        ):
            text = loc.get(key)
            if text:
                planned[col] = _us(near_utc(clock_secs(text), anchor))
        base = {
            "rid": rid,
            "ssd": ssd,
            "tpl": loc.get("tpl"),
            "loc_seq": seq,
            "plat": plat,
            "plat_src": plat_src,
            "plat_suppressed": plat_sup,
            "plat_confirmed": plat_conf,
            "length": length,
            "suppressed": suppressed,
            "loc_late_reason": loc_late,
            "ts_late_reason": ts_late,
            "update_origin": msg["origin"],
            "message_time": msg["us"],
            "src_file": src,
            **planned,
        }
        if not events:
            events = [(None, None)]
        for kind, ev in events:
            row = dict(base, event=kind)
            if ev is not None:
                for col, key in (
                    ("est", "et"),
                    ("actual", "at"),
                    ("est_min", "etmin"),
                    ("est_working", "wet"),
                ):
                    text = ev.get(key)
                    if text:
                        row[col] = _us(near_utc(clock_secs(text), anchor))
                row.update(
                    actual_class=ev.get("atClass"),
                    src=ev.get("src"),
                    src_inst=ev.get("srcInst"),
                    delayed=_bool(ev.get("delayed")),
                    est_unknown=_bool(ev.get("etUnknown")),
                    actual_removed=_bool(ev.get("atRemoved")),
                )
            sink.add("stop_updates", ssd, row)


def handle_association(el, msg, sink: Sink, src: str) -> None:
    main = next((c for c in el if local_name(c.tag) == "main"), None)
    assoc = next((c for c in el if local_name(c.tag) == "assoc"), None)
    if main is None or assoc is None:
        sink.counts["association_bad"] += 1
        return
    ssd = rid_ssd(main.get("rid") or "")
    ref = msg["us"] // 1_000_000
    if ssd is None:
        sink.counts["association_bad"] += 1
        return
    first = main.get("wta") or main.get("wtd") or main.get("pta") or main.get("ptd")
    anchor = service_day_times(ssd.toordinal(), [first], ref)[0] if first else ref

    def t(node, key):
        text = node.get(key)
        return _us(near_utc(clock_secs(text), anchor)) if text else None

    sink.add(
        "associations",
        ssd,
        {
            "main_rid": main.get("rid"),
            "assoc_rid": assoc.get("rid"),
            "ssd": ssd,
            "category": el.get("category"),
            "tiploc": el.get("tiploc"),
            "is_cancelled": _bool(el.get("isCancelled")) or False,
            "is_deleted": _bool(el.get("isDeleted")) or False,
            "main_wta": t(main, "wta"),
            "main_pta": t(main, "pta"),
            "assoc_wtd": t(assoc, "wtd"),
            "assoc_ptd": t(assoc, "ptd"),
            "message_time": msg["us"],
            "src_file": src,
        },
    )


def handle_deactivated(el, msg, sink: Sink, src: str) -> None:
    rid = el.get("rid")
    sink.add(
        "deactivations",
        rid_ssd(rid or ""),
        {"rid": rid, "ssd": rid_ssd(rid or ""), "message_time": msg["us"], "src_file": src},
    )


HANDLERS = {
    "schedule": handle_schedule,
    "TS": handle_ts,
    "association": handle_association,
    "deactivated": handle_deactivated,
}


# ---------------------------------------------------------------- one file


@dataclass
class FileStats:
    key: str
    status: str = "ok"  # ok, truncated, missing
    source_sha256: str | None = None
    gz_bytes: int = 0
    documents: int = 0
    parse_errors: int = 0
    message_kinds: dict[str, int] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    peak_rss_mb: float = 0.0
    error: str | None = None


def file_key(path: Path) -> str:
    return f"{path.parent.name}_{path.name.split('.')[0]}"


def peak_rss_mb() -> float:
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def parse_file(
    path: Path, out: Path, force: bool = False, cfg: "Stage1Config | None" = None
) -> FileStats:
    """Parse one hour file into Parquet parts under out/<table>/service_date=D/<key>.parquet.

    A missing file is logged and skipped. A truncated or corrupt gzip keeps every message read
    before the break and is marked truncated. A malformed message is counted and skipped.
    """
    log = get_logger("darwin_pport")
    key = file_key(path)
    stats = FileStats(key=key)
    stats_path = out / "_stats" / f"{key}.json"
    if stats_path.exists() and not force:
        return FileStats(**json.loads(stats_path.read_text()))
    started = time.monotonic()
    if not path.exists():
        stats.status = "missing"
        log.warning("archive_file_missing", path=str(path))
        stats.seconds = round(time.monotonic() - started, 2)
        return stats  # not cached, so a file mirrored later is picked up on the next run
    stats.gz_bytes = path.stat().st_size
    stats.source_sha256 = sha256_file(path)
    cfg = cfg or load_stage1_config()
    sink = Sink(out, key, stats.source_sha256, f"{path.parent.name}/{path.name}", cfg)
    kinds: Counter = Counter()
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    try:
        with gzip.open(path, "rb") as fh:
            for doc in iter_documents(fh):
                stats.documents += 1
                try:
                    root = etree.fromstring(doc, parser)
                    msg_us = parse_message_time(root.get("ts"))
                except (etree.XMLSyntaxError, TypeError, ValueError, AttributeError):
                    stats.parse_errors += 1
                    continue
                msg = {"us": msg_us[0], "ordinal": msg_us[1], "origin": None}
                for ur in root:
                    if local_name(ur.tag) != "uR":
                        kinds[local_name(ur.tag)] += 1
                        continue
                    msg["origin"] = ur.get("updateOrigin")
                    for el in ur:
                        name = local_name(el.tag)
                        kinds[name] += 1
                        handler = HANDLERS.get(name)
                        if handler is not None:
                            try:
                                handler(el, msg, sink, key)
                            except (ValueError, TypeError, AttributeError, KeyError):
                                stats.parse_errors += 1
    except (EOFError, gzip.BadGzipFile, zlib.error, OSError) as exc:
        stats.status = "truncated"
        stats.error = f"{type(exc).__name__}: {exc}"
        log.warning("archive_file_truncated", path=str(path), error=stats.error)
    sink.close()
    stats.message_kinds = dict(kinds)
    stats.rows = dict(sink.counts)
    return _finish(stats, stats_path, started)


def _finish(stats: FileStats, stats_path: Path, started: float) -> FileStats:
    stats.seconds = round(time.monotonic() - started, 2)
    stats.peak_rss_mb = round(peak_rss_mb(), 1)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = stats_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats.__dict__))
    tmp.rename(stats_path)
    return stats
