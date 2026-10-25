#!/usr/bin/env python3
"""Daily capture loss check against the community archive, for one UTC day.

    uv run python scripts/capture_loss.py [--day 2026-10-20] [--start D --end D] [--no-backfill]

With no day it checks the previous UTC day (the launchd timer runs it at 06:15 local). Archive
hours come from the read-only mirror in data/raw/darwin_archive when present, otherwise they are
downloaded one at a time into data/interim/loss_cache. Writes reports/capture_loss/DAY.json and
copies the archive files of short hours into data/raw/darwin_backfill/ (tagged
source=archive_backfill, never counted as captured).
"""

import argparse
import json
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from rail_risk.archive import (
    day_url,
    fetch_hour,
    local_path,
    parse_listing,
)
from rail_risk.capture_loss import (
    archive_files_for,
    backfill,
    scan_archive,
    scan_capture,
    summarise,
)
from rail_risk.logging import configure_logging, get_logger

MIRROR = Path("data/raw/darwin_archive")
CACHE = Path("data/interim/loss_cache")
CAPTURE = Path("data/raw/darwin")
BACKFILL = Path("data/raw/darwin_backfill")
REPORTS = Path("reports/capture_loss")
PAUSE = 2.0


def resolve_archive(day: date, log) -> tuple[dict[int, Path], list[str]]:
    """Local path per UTC hour, fetching what neither the mirror nor the cache holds."""
    wanted = archive_files_for(day)
    paths: dict[int, Path] = {}
    missing: list[str] = []
    listings: dict[date, dict[int, int] | None] = {}
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for hour, (ld, lh) in wanted.items():
            for root in (MIRROR, CACHE):
                if local_path(root, ld, lh).exists():
                    paths[hour] = local_path(root, ld, lh)
                    break
            if hour in paths:
                continue
            if ld not in listings:
                try:
                    resp = client.get(day_url(ld))
                    resp.raise_for_status()
                    listings[ld] = parse_listing(resp.text)
                except httpx.HTTPError as exc:
                    log.warning("archive_listing_failed", day=str(ld), error=str(exc))
                    listings[ld] = None
                time.sleep(PAUSE)
            hours = listings[ld]
            if not hours or lh not in hours:
                missing.append(f"{ld} {lh:02d}")
                continue
            dest = local_path(CACHE, ld, lh)
            try:
                fetch_hour(client, ld, lh, dest, hours[lh])
            except (httpx.HTTPError, OSError) as exc:
                log.warning("archive_fetch_failed", day=str(ld), hour=lh, error=str(exc))
                missing.append(f"{ld} {lh:02d}")
            else:
                paths[hour] = dest
            time.sleep(PAUSE)
    return paths, missing


def check_day(day: date, do_backfill: bool, log) -> dict:
    paths, missing = resolve_archive(day, log)
    arch = scan_archive(paths.values(), day)
    cap = scan_capture(CAPTURE, day)
    report = summarise(day, arch, cap)
    report["archive_files_missing"] = missing
    report["status"] = "ok" if not missing else "archive_incomplete"
    report["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    # a hour the archive could not supply cannot be called lost, so it is not backfilled
    report["backfilled"] = (
        backfill(report["gap_hours"], day, paths, BACKFILL) if do_backfill else []
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{day.isoformat()}.json").write_text(json.dumps(report, indent=1) + "\n")
    log.info(
        "capture_loss",
        day=day.isoformat(),
        loss_pct=report["loss_pct"],
        rid_loss_pct=report["rid_loss_pct"],
        duplicate_pct=report["duplicate_pct"],
        gap_hours=len(report["gap_hours"]),
        status=report["status"],
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=date.fromisoformat)
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--no-backfill", action="store_true")
    args = ap.parse_args()
    configure_logging("INFO", False)
    log = get_logger("capture_loss")
    if args.start and args.end:
        days = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]
    else:
        days = [args.day or datetime.now(UTC).date() - timedelta(days=1)]
    worst = 0
    for d in days:
        r = check_day(d, not args.no_backfill, log)
        worst = max(worst, 0 if r["status"] == "ok" else 2)
    return worst


if __name__ == "__main__":
    sys.exit(main())
