#!/usr/bin/env python3
"""Extract the mirrored Push Port hours to Parquet under data/interim/pport/.

    uv run python scripts/run_pport_extract.py --start 2026-10-10 --end 2026-10-16
    uv run python scripts/run_pport_extract.py --window            # the full project window
Finished hours are skipped on restart (a stats file per hour). Never decompresses to disk.
"""

import argparse
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path

from rail_risk.ingest.darwin_pport import load_stage1_config
from rail_risk.ingest.pport_extract import (
    archive_files,
    build_stop_actuals,
    days_between,
    extract,
    partial_days,
    split_counts,
)

ROOT = Path("data/raw/darwin_archive")
OUT = Path("data/interim/pport")
MIN_FREE_GB = 60


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--window", action="store_true", help="whole project window and boundary days")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_stage1_config()
    if args.window:
        days = sorted(
            {d for s, e in cfg.windows for d in days_between(s, e)}
            | set(cfg.boundary_days)
            | set(cfg.clock_change_days)
        )
    else:
        days = days_between(args.start, args.end)
    if (
        shutil.disk_usage(args.out.parent if args.out.parent.exists() else ".").free
        < MIN_FREE_GB * 1024**3
    ):
        print("free disk below floor", file=sys.stderr)
        return 3
    started = time.monotonic()
    files = [p for p in archive_files(ROOT, days)]
    results = extract(files, args.out, workers=args.workers, force=args.force)
    build_stop_actuals(args.out, days)
    secs = time.monotonic() - started
    size = sum(f.stat().st_size for f in args.out.rglob("*.parquet"))
    summary = {
        "days": len(days),
        "files": len(results),
        "seconds": round(secs, 1),
        "gb_on_disk": round(size / 1e9, 3),
        "rows": {
            t: sum(r.rows.get(t, 0) for r in results)
            for t in (
                "schedules",
                "schedule_points",
                "stop_updates",
                "associations",
                "deactivations",
            )
        },
        "peak_rss_mb_max": max((r.peak_rss_mb for r in results), default=0),
        "not_ok": partial_days(args.out / "_stats"),
        "parse_errors": sum(r.parse_errors for r in results),
        "splits": split_counts(args.out),
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
