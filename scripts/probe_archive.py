#!/usr/bin/env python3
"""Probe the community Darwin archive: fetch one hour or one day, then describe what came back.

    uv run python scripts/probe_archive.py 2026-10-16 --hour 1      # one hour
    uv run python scripts/probe_archive.py 2026-10-16               # the full day
Files go to data/raw/darwin_archive/ (write-once).
The manifest is data/interim/archive_manifest.sqlite.
"""

import argparse
import gzip
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path

import httpx

from rail_risk.archive import Manifest, local_path, mirror

ROOT = Path("data/raw/darwin_archive")
MANIFEST = Path("data/interim/archive_manifest.sqlite")
TAG = re.compile(
    rb"<(?:[A-Za-z0-9]+:)?(Pport|uR|sR|schedule|deactivated|association|TS|OW|trackingID|alarm|"
    rb"trainAlert|trainOrder|tracksIDs|scheduleFormations|formationLoading|FormationLoading)\b"
)


def describe(path: Path) -> dict:
    raw = path.read_bytes()
    text = gzip.decompress(raw)
    first = text[:600].decode("utf-8", "replace")
    kinds = Counter(m.group(1).decode() for m in TAG.finditer(text))
    ns = sorted(set(re.findall(rb'xmlns(?::\w+)?="([^"]+)"', text[:200000])))
    return {
        "gz_bytes": len(raw),
        "raw_bytes": len(text),
        "lines": text.count(b"\n") + 1,
        "pport_messages": kinds.get("Pport", 0),
        "kinds": dict(kinds),
        "version": sorted(set(re.findall(rb'version="([^"]+)"', text[:200000])))[:5],
        "namespaces": [n.decode() for n in ns],
        "head": first,
    }


def list_window(start: date, end: date) -> None:
    """One page per day, newest first, one at a time; saves listings to the manifest."""
    from rail_risk.archive import date_range_newest_first, day_url, parse_listing

    manifest = Manifest(MANIFEST)
    files = total = missing = days_empty = 0
    with httpx.Client(
        timeout=60, follow_redirects=True, headers={"User-Agent": "rail-risk-probe"}
    ) as c:
        for day in date_range_newest_first(start, end):
            hours = manifest.listing(day)
            if hours is None:
                hours = parse_listing(c.get(day_url(day)).text)
                manifest.save_listing(day, hours)
                time.sleep(1.0)
            files += len(hours)
            total += sum(hours.values())
            missing += 24 - len(hours)
            days_empty += not hours
    print(
        f"{start}..{end}: files={files} compressed_gb={total / 1e9:.2f} missing_hours={missing}"
        f" empty_days={days_empty}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("day", type=date.fromisoformat, nargs="?")
    ap.add_argument("--hour", type=int)
    ap.add_argument(
        "--listings",
        nargs=2,
        metavar=("START", "END"),
        type=date.fromisoformat,
        help="fetch only the day listings (no files) and print the size estimate",
    )
    args = ap.parse_args()
    if args.listings:
        list_window(*args.listings)
        return
    manifest = Manifest(MANIFEST)
    with httpx.Client(
        timeout=120, follow_redirects=True, headers={"User-Agent": "rail-risk-probe"}
    ) as c:
        started = time.monotonic()
        if args.hour is None:
            res = mirror([args.day], ROOT, manifest, c, pause=2.0)
        else:
            listing = manifest.listing(args.day)
            if listing is None:
                from rail_risk.archive import day_url, parse_listing

                listing = parse_listing(c.get(day_url(args.day)).text)
                manifest.save_listing(args.day, listing)
            from rail_risk.archive import fetch_hour

            dest = local_path(ROOT, args.day, args.hour)
            n, sha = fetch_hour(c, args.day, args.hour, dest, listing.get(args.hour))
            manifest.record(args.day, args.hour, "ok", n, listing.get(args.hour), sha)
            res = None
        secs = time.monotonic() - started
    print(f"result={res} seconds={secs:.1f}")
    for hour in range(24):
        p = local_path(ROOT, args.day, hour)
        if p.exists():
            d = describe(p)
            head = d.pop("head")
            print(hour, d)
            if args.hour is not None:
                print(head)


if __name__ == "__main__":
    main()
