#!/usr/bin/env python3
"""Compare one archive day with our own Darwin capture by message-timestamp hour and by rid.

Both sources are filed by the hour written in the message timestamp, which carries the UK offset;
our capture files are named by UTC hour, so each message is bucketed by its own timestamp here.
    uv run python scripts/compare_archive_capture.py 2026-10-16
"""

import argparse
import gzip
import json
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ARCHIVE = Path("data/raw/darwin_archive")
CAPTURE = Path("data/raw/darwin")
TS = re.compile(rb'(?:"ts":"|\bts=")(\d{4}-\d\d-\d\d)T(\d\d)')
RID = re.compile(rb'(?:"rid":"|\brid=")(\d{15})')


def scan(paths: list[Path], day: date) -> tuple[Counter, dict[int, set]]:
    msgs: Counter = Counter()
    rids: dict[int, set] = {}
    want = day.isoformat().encode()
    for p in paths:
        if not p.exists():
            continue
        with gzip.open(p, "rb") as fh:
            for line in fh:
                m = TS.search(line)
                if not m or m.group(1) != want:
                    continue
                h = int(m.group(2))
                msgs[h] += 1
                rids.setdefault(h, set()).update(r.decode() for r in RID.findall(line))
    return msgs, rids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("day", type=date.fromisoformat)
    args = ap.parse_args()
    d = args.day
    prev = d - timedelta(days=1)
    arch_files = [ARCHIVE / d.isoformat() / f"{h:02d}.pport.gz" for h in range(24)]
    cap_files = [CAPTURE / prev.isoformat() / "23.jsonl.gz"] + [
        CAPTURE / d.isoformat() / f"{h:02d}.jsonl.gz" for h in range(24)
    ]
    a_msgs, a_rids = scan(arch_files, d)
    c_msgs, c_rids = scan(cap_files, d)
    out = {"day": d.isoformat(), "hours": {}}
    for h in range(24):
        ratio = c_msgs[h] / a_msgs[h] if a_msgs[h] else None
        both = c_rids.get(h, set()) & a_rids.get(h, set())
        share = len(both) / len(c_rids[h]) if c_rids.get(h) else None
        out["hours"][h] = {
            "archive_msgs": a_msgs[h],
            "capture_msgs": c_msgs[h],
            "ratio": None if ratio is None else round(ratio, 3),
            "capture_rids": len(c_rids.get(h, ())),
            "archive_rids": len(a_rids.get(h, ())),
            "capture_rids_in_archive": None if share is None else round(share, 4),
        }
    all_a = set().union(*a_rids.values()) if a_rids else set()
    all_c = set().union(*c_rids.values()) if c_rids else set()
    out["day_totals"] = {
        "archive_msgs": sum(a_msgs.values()),
        "capture_msgs": sum(c_msgs.values()),
        "capture_rids": len(all_c),
        "archive_rids": len(all_a),
        "capture_rids_in_archive": round(len(all_c & all_a) / len(all_c), 4) if all_c else None,
        "archive_rids_in_capture": round(len(all_c & all_a) / len(all_a), 4) if all_a else None,
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
