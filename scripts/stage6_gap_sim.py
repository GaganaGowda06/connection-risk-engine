#!/usr/bin/env python3
# ruff: noqa: E501
"""Simulated feed outages for the guard comparison (train_pool days only: never test or oot_week).

    uv run python scripts/stage6_gap_sim.py make                 # writes reports/serving/sim_gaps.json
    uv run python scripts/stage6_gap_sim.py replay --block validation --max-hours 48   # resumable

`make` fixes 28 message windows (30 to 240 minutes) before any replay: two slots per UTC day (one
starting 06:00 to 09:00, one 17:00 to 20:00), so each window has at least four hours of clean feed
before the next one; 22 of 28 slots on the 14 validation days, 6 of 8 on four train days. `replay`
runs the engine hour by hour with those windows removed from the messages (the guard is applied
afterwards, from the ledger), in one state directory per block, resuming where it stopped.
"""

import argparse
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.serving import engine as E
from rail_risk.serving import state as S

OUT = Path("reports/serving/sim_gaps.json")
WORK = Path("data/interim/serve/gapsim_v2")
BLOCKS = {
    "validation": (date(2026, 9, 25), date(2026, 10, 8), 22),
    "train": (date(2026, 9, 19), date(2026, 9, 22), 6),
}
DURATIONS = [30, 60, 90, 120, 180, 240]
SEED = 20261026


def make() -> dict:
    rng = np.random.default_rng(SEED)
    gaps = []
    for block, (d0, d1, n) in BLOCKS.items():
        days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
        slots = [(d, s) for d in days for s in ("am", "pm")]
        pick = sorted(rng.choice(len(slots), size=n, replace=False))
        dur = (DURATIONS * (n // len(DURATIONS) + 1))[:n]
        rng.shuffle(dur)
        for k, i in enumerate(pick):
            d, s = slots[i]
            base = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=6 if s == "am" else 17)
            start = base + pd.Timedelta(minutes=int(rng.integers(0, 37)) * 5)
            end = start + pd.Timedelta(minutes=int(dur[k]))
            gaps.append({"block": block, "day": str(d), "weekday": d.strftime("%a"), "slot": s,
                         "start": str(start), "end": str(end), "minutes": int(dur[k]),
                         "half": "fit" if days.index(d) % 2 == 0 else "score"})  # fmt: skip
    res = {"seed": SEED, "recovery_window_min": 240, "gaps": gaps}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    return res


def load_gaps(block: str | None = None) -> list[dict]:
    g = json.loads(OUT.read_text())["gaps"]
    return [x for x in g if block is None or x["block"] == block]


def replay(block: str, max_hours: int, fresh: bool) -> None:
    d0, d1, _ = BLOCKS[block]
    work = WORK / block
    if fresh and work.exists():
        shutil.rmtree(work)
    gaps = [(pd.Timestamp(g["start"]), pd.Timestamp(g["end"])) for g in load_gaps(block)]
    ssds = [d0 + timedelta(days=i) for i in range(-1, (d1 - d0).days + 3)]
    start = pd.Timestamp(d0, tz="UTC")
    end = pd.Timestamp(d1 + timedelta(days=1), tz="UTC")
    th = E.Thresholds.load(Path("reports/serving/guard_threshold.json"))
    store = S.open_store("sqlite", work / "state")
    eng = E.Engine(store, Path("data/interim/pport"), work / "store", th, ssds=ssds, gaps=gaps)
    done = {pd.Timestamp(h) for h in store.hours_done()}
    if not done:
        eng.preload(start, ssds)
    n, h = 0, start
    while h < end and n < max_hours:
        if h not in done:
            st = eng.process_hour(h)
            n += 1
            print(st["hour"], st["arrival_rows"], st["pair_rows"], st["peak_mb"], flush=True)
        h += pd.Timedelta(hours=1)
    left = (
        sum(
            1
            for i in range(int((end - start) / pd.Timedelta(hours=1)))
            if (start + pd.Timedelta(hours=i)) not in done
        )
        - n
    )
    print(f"REMAINING {left}", flush=True)
    store.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "replay"])
    ap.add_argument("--block", choices=list(BLOCKS))
    ap.add_argument("--max-hours", type=int, default=48)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    if a.cmd == "make":
        r = make()
        print(pd.DataFrame(r["gaps"]).to_string())
    else:
        replay(a.block, a.max_hours, a.fresh)


if __name__ == "__main__":
    main()
