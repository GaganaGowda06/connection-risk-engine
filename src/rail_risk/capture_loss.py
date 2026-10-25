"""Daily loss check: our Kafka capture against the community archive for one UTC day.

Both sources are scanned with regexes (no JSON or XML parse). Each message is bucketed by the
UTC hour of its own `ts`, so the hour naming of the files (UTC for ours, UK local for the
archive) does not matter. Capture lines are de-duplicated on (partition, offset) first, so a
replay is counted as a duplicate and never as captured data. Hours the capture is short on are
copied from the archive into a separate backfill directory and tagged; backfill is never
counted as captured.
"""

import gzip
import json
import re
import shutil
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
TS = re.compile(rb'(?:"ts":"|\bts=")(\d{4}-\d\d-\d\d)T(\d\d):\d\d:\d\d[\d.]*([+-]\d\d):\d\d')
RID = re.compile(rb'(?:"rid":"|\brid=")(\d{15})')
OFFSET = re.compile(rb'"partition":(\d+),"offset":(\d+)')
SHORT = 0.99  # an hour is a gap hour when the capture holds under this share of the archive


def archive_file(day: date, hour: int) -> tuple[date, int]:
    """The (local date, local hour) archive file that holds a UTC hour."""
    local = datetime(day.year, day.month, day.day, hour, tzinfo=UTC).astimezone(LONDON)
    return local.date(), local.hour


def archive_files_for(day: date) -> dict[int, tuple[date, int]]:
    return {h: archive_file(day, h) for h in range(24)}


def utc_hour_of(date_s: bytes, hour_s: bytes, offset_s: bytes, cache: dict) -> tuple[date, int]:
    key = (date_s, hour_s, offset_s)
    got = cache.get(key)
    if got is None:
        base = datetime.fromisoformat(date_s.decode()) + timedelta(
            hours=int(hour_s) - int(offset_s)
        )
        got = cache[key] = (base.date(), base.hour)
    return got


class Scan:
    """Per UTC hour message counts and rid sets for one UTC day, from one kind of source."""

    def __init__(self, day: date) -> None:
        self.day = day
        self.msgs: Counter[int] = Counter()
        self.rids: dict[int, set[bytes]] = {h: set() for h in range(24)}
        self.lines = 0
        self.duplicates = 0
        self._cache: dict = {}

    def add_file(self, path: Path, dedupe_offsets: set | None = None) -> None:
        with gzip.open(path, "rb") as fh:
            try:
                for line in fh:
                    self.lines += 1
                    m = TS.search(line)
                    if not m:
                        continue
                    if dedupe_offsets is not None:
                        o = OFFSET.search(line[:300])
                        if o:
                            key = (int(o.group(1)), int(o.group(2)))
                            if key in dedupe_offsets:
                                self.duplicates += 1
                                continue
                            dedupe_offsets.add(key)
                    d, h = utc_hour_of(m.group(1), m.group(2), m.group(3), self._cache)
                    if d != self.day:
                        continue
                    self.msgs[h] += 1
                    self.rids[h].update(RID.findall(line))
            except (EOFError, OSError, gzip.BadGzipFile):
                pass  # a truncated file keeps what was read; the hour shows as short


def scan_capture(capture_root: Path, day: date) -> Scan:
    prev, nxt = day - timedelta(days=1), day + timedelta(days=1)
    paths = [capture_root / prev.isoformat() / "23.jsonl.gz"]
    paths += [capture_root / day.isoformat() / f"{h:02d}.jsonl.gz" for h in range(24)]
    paths += [capture_root / nxt.isoformat() / "00.jsonl.gz"]
    scan = Scan(day)
    seen: set[tuple[int, int]] = set()
    for p in paths:
        if p.exists():
            scan.add_file(p, seen)
    return scan


def scan_archive(files: Iterable[Path], day: date) -> Scan:
    scan = Scan(day)
    for p in dict.fromkeys(files):
        if p.exists():
            scan.add_file(p)
    return scan


def hour_report(arch: Scan, cap: Scan) -> list[dict]:
    rows = []
    for h in range(24):
        a, c = arch.msgs[h], cap.msgs[h]
        missing = arch.rids[h] - cap.rids[h]
        rows.append(
            {
                "hour_utc": h,
                "archive_msgs": a,
                "capture_msgs": c,
                "msg_ratio": round(c / a, 4) if a else None,
                "archive_rids": len(arch.rids[h]),
                "rids_missing": len(missing),
                "rid_loss_pct": round(100 * len(missing) / len(arch.rids[h]), 3)
                if arch.rids[h]
                else None,
            }
        )
    return rows


def summarise(day: date, arch: Scan, cap: Scan, expected_hours: int = 24) -> dict:
    rows = hour_report(arch, cap)
    a_total = sum(r["archive_msgs"] for r in rows)
    lost = sum(max(0, r["archive_msgs"] - r["capture_msgs"]) for r in rows)
    a_rids = sum(r["archive_rids"] for r in rows)
    gap = [r["hour_utc"] for r in rows if r["msg_ratio"] is not None and r["msg_ratio"] < SHORT]
    return {
        "day_utc": day.isoformat(),
        "archive_msgs": a_total,
        "capture_msgs": sum(r["capture_msgs"] for r in rows),
        "loss_pct": round(100 * lost / a_total, 3) if a_total else None,
        "rid_loss_pct": round(100 * sum(r["rids_missing"] for r in rows) / a_rids, 3)
        if a_rids
        else None,
        "capture_lines": cap.lines,
        "duplicate_lines": cap.duplicates,
        "duplicate_pct": round(100 * cap.duplicates / cap.lines, 3) if cap.lines else None,
        "gap_hours": gap,
        "empty_hours": [
            r["hour_utc"] for r in rows if r["capture_msgs"] == 0 and r["archive_msgs"]
        ],
        "archive_hours_without_messages": [r["hour_utc"] for r in rows if r["archive_msgs"] == 0],
        "hours": rows,
    }


def backfill(
    gap_hours: list[int], day: date, archive_paths: dict[int, Path], out_root: Path
) -> list[dict]:
    """Copy the archive file of each gap hour to data/raw/darwin_backfill/DAY/HH.pport.gz.

    The files stay apart from data/raw/darwin and carry a sidecar tag, so backfill can never be
    mistaken for captured data.
    """
    written = []
    for h in gap_hours:
        src = archive_paths.get(h)
        if src is None or not src.exists():
            continue
        dest = out_root / day.isoformat() / f"{h:02d}.pport.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        tag = {"source": "archive_backfill", "hour_utc": h, "archive_file": src.name}
        dest.with_name(dest.name + ".source.json").write_text(json.dumps(tag))
        written.append({"hour_utc": h, "path": str(dest), "source": "archive_backfill"})
    return written
