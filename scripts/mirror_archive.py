#!/usr/bin/env python3
"""Resumable mirror of the community Darwin archive for the project windows, newest first.

One request at a time with a pause between files, retries with backoff, a manifest with sizes and
sha256, and a hard stop when free disk falls below the floor. Safe to stop and restart.
    uv run python scripts/mirror_archive.py [--pause 2] [--min-free-gb 60]
"""

import argparse
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from rail_risk.archive import Manifest, date_range_newest_first, mirror
from rail_risk.ingest.darwin_pport import load_stage1_config

ROOT = Path("data/raw/darwin_archive")
MANIFEST = Path("data/interim/archive_manifest.sqlite")
WINDOWS = [(date(2026, 7, 25), date(2026, 10, 23)), (date(2025, 12, 1), date(2026, 2, 28))]


def stamp(msg: str) -> None:
    print(f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--min-free-gb", type=float, default=60.0)
    ap.add_argument(
        "--extra",
        action="store_true",
        help="only the boundary and clock-change days of config/stage1.yaml",
    )
    args = ap.parse_args()
    windows = WINDOWS
    if args.extra:
        cfg = load_stage1_config()
        windows = [
            (d, d) for d in sorted({*cfg.boundary_days, *cfg.clock_change_days}, reverse=True)
        ]
    manifest = Manifest(MANIFEST)
    stamp(f"mirror start windows={windows}")
    headers = {"User-Agent": "rail-risk-mirror (research, one request at a time)"}
    with httpx.Client(timeout=180, follow_redirects=True, headers=headers) as client:
        for start, end in windows:
            res = mirror(
                date_range_newest_first(start, end),
                ROOT,
                manifest,
                client,
                pause=args.pause,
                min_free_gb=args.min_free_gb,
                log=stamp,
            )
            stamp(f"window {start}..{end} {res}")
            if res.stopped_low_disk:
                return 3
    failed = manifest.db.execute("SELECT COUNT(*) FROM files WHERE status='failed'").fetchone()[0]
    stamp(f"mirror finished counts={manifest.counts()} gaps={manifest.gaps()}")
    return 2 if failed else 0  # 2 means some hours failed; rerun to retry them


if __name__ == "__main__":
    raise SystemExit(main())
