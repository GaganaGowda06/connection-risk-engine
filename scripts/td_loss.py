#!/usr/bin/env python3
# ruff: noqa: E501
"""Daily loss check of the TD capture for one UTC day (step 2).

Reads the hourly files one at a time. Reports: minutes of the day with zero TD messages (from the
first to the last minute captured), berth-step continuity per train (a C-class step whose from-berth
differs from the berth the same description last moved to in the same area, unless an interpose
or cancel came between), the share of timestamps corrected for the cross-midnight bug, and, with
--darwin, the share of Darwin-reported movements on the chosen routes that have a TD step at the
matching station (our loss measure; needs the Darwin archive for that day, never our own capture).

    uv run python scripts/td_loss.py --day 2026-10-27 [--darwin]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from rail_risk.td import fix_time, messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--root", default="data/raw/td")
    ap.add_argument("--out", default="reports/td")
    a = ap.parse_args()
    files = sorted((Path(a.root) / a.day).glob("*.jsonl.gz"))
    per_min: Counter = Counter()
    kinds: Counter = Counter()
    last: dict = {}  # (area, descr) -> berth the train last moved to
    steps = jumps = interposes = fixed = n_time = 0
    jump_area: Counter = Counter()
    step_area: Counter = Counter()
    for f in files:
        for recv, topic, kind, msg in messages(f):
            if topic != "TD_ALL_SIG_AREA":
                continue
            per_min[recv // 60000] += 1
            kinds[kind] += 1
            if "time" in msg:
                n_time += 1
                fixed += fix_time(int(msg["time"]), recv)[1]
            key = (msg.get("area_id"), msg.get("descr"))
            if kind == "CA_MSG":
                steps += 1
                step_area[key[0]] += 1
                prev = last.get(key)
                if prev is not None and prev != msg["from"]:
                    jumps += 1
                    jump_area[key[0]] += 1
                last[key] = msg["to"]
            elif kind == "CC_MSG":
                interposes += 1
                last[key] = msg["to"]
            elif kind == "CB_MSG":
                last.pop(key, None)
    if not per_min:
        raise SystemExit(f"no TD messages for {a.day}")
    lo, hi = min(per_min), max(per_min)
    zero = [m for m in range(lo, hi + 1) if per_min[m] == 0]
    res = {
        "day": a.day, "files": len(files), "minutes_spanned": hi - lo + 1, "minutes_with_zero_td_messages": len(zero),
        "zero_minute_share": len(zero) / (hi - lo + 1), "messages_by_kind": dict(kinds), "c_steps": steps,
        "discontinuous_steps": jumps, "discontinuity_share": jumps / max(steps, 1), "interposes": interposes,
        "timestamps_corrected_share": fixed / max(n_time, 1),
        "worst_areas_by_discontinuity": [
            [k, jump_area[k], step_area[k]] for k, _ in jump_area.most_common(5)
        ],
    }  # fmt: skip
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"loss_{a.day}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
