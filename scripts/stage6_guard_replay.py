#!/usr/bin/env python3
# ruff: noqa: E501
"""Validation replay for the freshness guard (train_pool days only, never oot_week).

Replays 2026-10-01 to 2026-10-04 hour by hour through the engine with the guard off, with two
archive hours removed the way the real archive lost them (2026-10-02 09:00 to 10:00 UTC and
2026-10-03 15:00 to 17:00 UTC), and joins the true outcomes afterwards. The ledger keeps every
freshness signal so thresholds can be chosen offline (scripts/stage6_guard_choose.py).

    uv run python scripts/stage6_guard_replay.py
"""

import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from rail_risk.serving import engine as E
from rail_risk.serving import state as S
from rail_risk.serving.replay_gaps import GAPS

WORK = Path("data/interim/serve/guard_replay")
SSDS = [
    date(2026, 9, 30),
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 3),
    date(2026, 10, 4),
]
START = pd.Timestamp("2026-10-01 00:00", tz="UTC")
END = pd.Timestamp("2026-10-05 00:00", tz="UTC")


def main() -> None:
    if WORK.exists() and "--fresh" in sys.argv:
        shutil.rmtree(WORK)
    store = S.open_store("sqlite", WORK / "state")
    eng = E.Engine(store, Path("data/interim/pport"), WORK / "store", ssds=SSDS)
    for a, b in GAPS:
        h = a
        while h < b:
            eng.skip.add(h)
            h += pd.Timedelta(hours=1)
    done = set(store.hours_done())
    if not done:
        eng.preload(START, SSDS)
    h = START
    while h < END:
        if str(h) not in done:
            st = eng.process_hour(h)
            print(
                st["hour"],
                st["arrival_rows"],
                st["pair_rows"],
                st["peak_mb"],
                st["seconds"],
                flush=True,
            )
        h += pd.Timedelta(hours=1)
    store.close()


if __name__ == "__main__":
    main()
