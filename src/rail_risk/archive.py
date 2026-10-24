"""Mirror of the community Darwin Push Port archive (ilovetrains.co.uk), one file at a time.

Files are hourly gzips named by the hour in the message timestamp. The day page lists the hours
that exist with their sizes, so an hour absent from the listing is recorded as a gap rather than
requested. Raw files are write-once; the manifest (sqlite) makes the mirror resumable.
"""

import hashlib
import json
import re
import shutil
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import httpx

BASE = "https://ilovetrains.co.uk"
LISTING_ITEM = re.compile(r'"name":"[^"]+","hour":(\d+),"downloadUrl":"[^"]+","sizeBytes":(\d+)')
GB = 1024**3


def day_url(day: date) -> str:
    return f"{BASE}/darwin-push-port/{day.isoformat()}"


def hour_path(day: date, hour: int) -> str:
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be 0 to 23, got {hour}")
    return f"{day:%Y/%m/%d}/{hour:02d}.pport.gz"


def download_url(day: date, hour: int) -> str:
    return f"{BASE}/api/download-darwin-dump?fileName={quote(hour_path(day, hour), safe='')}"


def local_path(root: Path, day: date, hour: int) -> Path:
    return root / day.isoformat() / f"{hour:02d}.pport.gz"


def parse_listing(html: str) -> dict[int, int]:
    """Hour to compressed size in bytes, from the day page (the listing is in its flight data)."""
    text = html.replace('\\"', '"')
    return {int(h): int(size) for h, size in LISTING_ITEM.findall(text)}


def date_range_newest_first(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end before start")
    return [end - timedelta(days=i) for i in range((end - start).days + 1)]


class Manifest:
    """One row per (date, hour): fetched with size and sha256, or a recorded gap."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS files (day TEXT, hour INTEGER, status TEXT, bytes INTEGER,"
            " listed_bytes INTEGER, sha256 TEXT, fetched_at TEXT, PRIMARY KEY (day, hour))"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS listings (day TEXT PRIMARY KEY, hours TEXT)")
        self.db.commit()

    def listing(self, day: date) -> dict[int, int] | None:
        row = self.db.execute(
            "SELECT hours FROM listings WHERE day=?", (day.isoformat(),)
        ).fetchone()
        return None if row is None else {int(h): s for h, s in json.loads(row[0]).items()}

    def save_listing(self, day: date, hours: dict[int, int]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO listings VALUES (?, ?)", (day.isoformat(), json.dumps(hours))
        )
        self.db.commit()

    def status(self, day: date, hour: int) -> str | None:
        row = self.db.execute(
            "SELECT status FROM files WHERE day=? AND hour=?", (day.isoformat(), hour)
        ).fetchone()
        return row[0] if row else None

    def record(
        self,
        day: date,
        hour: int,
        status: str,
        nbytes: int | None = None,
        listed: int | None = None,
        sha256: str | None = None,
    ) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.db.execute(
            "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
            (day.isoformat(), hour, status, nbytes, listed, sha256, stamp),
        )
        self.db.commit()

    def counts(self) -> dict[str, tuple[int, int]]:
        rows = self.db.execute(
            "SELECT status, COUNT(*), COALESCE(SUM(bytes), 0) FROM files GROUP BY status"
        ).fetchall()
        return {s: (n, b) for s, n, b in rows}

    def gaps(self) -> list[tuple[str, int]]:
        return self.db.execute(
            "SELECT day, hour FROM files WHERE status IN ('gap', 'failed') ORDER BY day, hour"
        ).fetchall()


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def fetch_hour(
    client: httpx.Client, day: date, hour: int, dest: Path, listed: int | None
) -> tuple[int, str]:
    """Stream to dest.part, check the listed size, rename; return (bytes, sha256)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    total = 0
    with client.stream("GET", download_url(day, hour)) as resp:
        resp.raise_for_status()
        with part.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
                digest.update(chunk)
                total += len(chunk)
    if listed is not None and total != listed:
        part.unlink()
        raise OSError(f"size {total} differs from listed {listed}")
    part.rename(dest)
    return total, digest.hexdigest()


@dataclass
class MirrorResult:
    fetched: int = 0
    skipped: int = 0
    gaps: int = 0
    failed: int = 0
    stopped_low_disk: bool = False


def mirror(
    days: Iterable[date],
    root: Path,
    manifest: Manifest,
    client: httpx.Client,
    pause: float = 2.0,
    min_free_gb: float = 60.0,
    retries: int = 3,
    backoff: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> MirrorResult:
    """Fetch every listed hour once, one request at a time, newest day first."""
    out = MirrorResult()
    root.mkdir(parents=True, exist_ok=True)
    for day in days:
        hours = manifest.listing(day)
        if hours is None:
            resp = _get_with_retries(client, day_url(day), retries, backoff, sleep)
            hours = parse_listing(resp.text) if resp is not None else None
            if hours is None:
                log(f"{day} listing failed; will retry on the next run")
                out.failed += 1
                continue
            manifest.save_listing(day, hours)
            sleep(pause)
        for hour in range(24):
            if manifest.status(day, hour) in ("ok", "gap"):
                out.skipped += 1
                continue
            if hour not in hours:
                manifest.record(day, hour, "gap")
                out.gaps += 1
                continue
            if free_bytes(root) < min_free_gb * GB:
                log(f"free disk below {min_free_gb} GB; stopping")
                out.stopped_low_disk = True
                return out
            dest = local_path(root, day, hour)
            for attempt in range(retries):
                try:
                    nbytes, sha = fetch_hour(client, day, hour, dest, hours[hour])
                except (httpx.HTTPError, OSError) as exc:
                    log(f"{day} {hour:02d} attempt {attempt + 1} failed: {exc}")
                    sleep(backoff * 2**attempt)
                else:
                    manifest.record(day, hour, "ok", nbytes, hours[hour], sha)
                    out.fetched += 1
                    log(f"{day} {hour:02d} ok {nbytes}")
                    break
            else:
                manifest.record(day, hour, "failed", listed=hours[hour])
                out.failed += 1
            sleep(pause)
    return out


def _get_with_retries(
    client: httpx.Client, url: str, retries: int, backoff: float, sleep: Callable[[float], None]
) -> httpx.Response | None:
    for attempt in range(retries):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp
        except httpx.HTTPError:
            sleep(backoff * 2**attempt)
    return None
