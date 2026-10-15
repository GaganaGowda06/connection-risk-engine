"""Read-only progress report for the HSP backfill and the Darwin capture. Prints no secrets."""

import sqlite3
import time
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


def darwin_status(root: Path) -> list[str]:
    files = sorted(root.glob("*/*.jsonl.gz"), key=lambda p: p.stat().st_mtime)
    if not files:
        return ["no capture files yet"]
    newest = files[-1]
    age_min = (time.time() - newest.stat().st_mtime) / 60
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    today_files = list((root / today).glob("*.jsonl.gz"))
    today_bytes = sum(p.stat().st_size for p in today_files)
    return [
        f"newest file: {newest.relative_to(root)}",
        f"age: {age_min:.1f} min since last write, size: {newest.stat().st_size / 1e6:.2f} MB",
        f"today (UTC) total: {today_bytes / 1e6:.1f} MB in {len(today_files)} files",
    ]
