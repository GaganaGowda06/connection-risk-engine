"""Read-only progress report for the HSP backfill and the Darwin capture. Prints no secrets."""

import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

WINDOWS_PER_FULL_DAY = 16  # hours 06-22


def hsp_status(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        return ["no manifest yet"]
    db = sqlite3.connect(f"file:{manifest_path}?mode=ro", uri=True)
    try:
        routes = [r[0] for r in db.execute("SELECT DISTINCT route FROM windows ORDER BY route")]
        if not routes:
            return ["no windows completed yet"]
        lines = []
        for route in routes:
            days = db.execute(
                "SELECT day, COUNT(*), SUM(services), SUM(rows), SUM(quarantined) "
                "FROM windows WHERE route=? GROUP BY day ORDER BY day",
                (route,),
            ).fetchall()
            complete = [d for d in days if d[1] >= WINDOWS_PER_FULL_DAY]
            calls = db.execute(
                "SELECT COALESCE(SUM(metrics_calls),0), COALESCE(SUM(detail_calls),0), "
                "COALESCE(SUM(retries),0), COALESCE(SUM(failed),0), COUNT(*) "
                "FROM call_log WHERE route=?",
                (route,),
            ).fetchone()
            rows = sum(d[3] for d in days)
            quarantined = sum(d[4] for d in days)
            span = f"{complete[0][0]} to {complete[-1][0]}" if complete else "none"
            lines.append(
                f"{route}: {len(complete)} complete days ({span}), "
                f"{len(days) - len(complete)} partial, rows={rows}, quarantined={quarantined}"
            )
            lines.append(
                f"  calls since logging began: metrics={calls[0]} details={calls[1]} "
                f"retries={calls[2]} failed_windows={calls[3]} windows_logged={calls[4]}"
            )
        return lines
    finally:
        db.close()


STALE_PARSE_HOURS = 2.0


@dataclass(frozen=True)
class ParseLag:
    """How far the parsed Darwin tables trail the raw capture, in whole hourly files."""

    newest_raw: str | None  # "YYYY-MM-DD/HH"
    newest_parsed: str | None
    lag_hours: float | None
    stale: bool


def _key_time(key: str) -> datetime:
    day, hour = key.split("/")
    return datetime.fromisoformat(day).replace(hour=int(hour), tzinfo=UTC)


def parse_lag(raw_root: Path, manifest: Path) -> ParseLag:
    """The newest raw hourly file against the newest parsed one.

    The current hour is never parsed, so a healthy lag is about one hour. More than
    STALE_PARSE_HOURS behind means closed hours are waiting to be parsed, and anything that
    reads only the parsed tables is missing their messages, schedules included. No raw files
    is not stale. Raw files with nothing parsed at all is stale.
    """
    raw = sorted(f"{p.parent.name}/{p.name.split('.')[0]}" for p in raw_root.glob("*/*.jsonl.gz"))
    if not raw:
        return ParseLag(None, None, None, False)
    newest_raw = max(raw, key=_key_time)
    parsed: list[str] = []
    if manifest.exists():
        db = sqlite3.connect(f"file:{manifest}?mode=ro", uri=True)
        try:
            parsed = [r[0] for r in db.execute("SELECT key FROM files")]
        finally:
            db.close()
    parsed = [k for k in parsed if "/" in k]
    if not parsed:
        return ParseLag(newest_raw, None, None, True)
    newest_parsed = max(parsed, key=_key_time)
    lag = (_key_time(newest_raw) - _key_time(newest_parsed)).total_seconds() / 3600.0
    return ParseLag(newest_raw, newest_parsed, lag, lag > STALE_PARSE_HOURS)


def darwin_status(root: Path, interim: Path | None = None) -> list[str]:
    files = sorted(root.glob("*/*.jsonl.gz"), key=lambda p: p.stat().st_mtime)
    if not files:
        return ["no capture files yet"]
    newest = files[-1]
    age_min = (time.time() - newest.stat().st_mtime) / 60
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    today_files = list((root / today).glob("*.jsonl.gz"))
    today_bytes = sum(p.stat().st_size for p in today_files)
    lines = [
        f"newest file: {newest.relative_to(root)}",
        f"age: {age_min:.1f} min since last write, size: {newest.stat().st_size / 1e6:.2f} MB",
        f"today (UTC) total: {today_bytes / 1e6:.1f} MB in {len(today_files)} files",
    ]
    manifest = (interim or Path()) / "darwin_manifest.sqlite"
    if interim is not None and manifest.exists():
        db = sqlite3.connect(f"file:{manifest}?mode=ro", uri=True)
        try:
            n, sched, events, assoc, quar, dups, lost, last = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(schedules),0), COALESCE(SUM(events),0), "
                "COALESCE(SUM(associations),0), "
                "COALESCE(SUM(q_schedules+q_events+q_associations+q_messages),0), "
                "COALESCE(SUM(duplicates),0), COALESCE(SUM(bytes_lost),0), MAX(parsed_at) "
                "FROM files"
            ).fetchone()
        finally:
            db.close()
        lines.append(
            f"parsed: {n} files, schedules={sched}, events={events}, associations={assoc}, "
            f"quarantined={quar}, duplicates_removed={dups}, bytes_lost={lost}, last={last}"
        )
    else:
        lines.append("parsed: nothing yet (run rail-risk parse-darwin)")
    lag = parse_lag(root, manifest)
    if lag.newest_raw is not None:
        how = "nothing parsed" if lag.lag_hours is None else f"{lag.lag_hours:.0f} h behind"
        lines.append(
            f"parse lag: newest raw {lag.newest_raw}, newest parsed {lag.newest_parsed}, {how}"
            + (f"  STALE (more than {STALE_PARSE_HOURS:.0f} h; run rail-risk parse-darwin)"
               if lag.stale else "")
        )  # fmt: skip
    return lines
