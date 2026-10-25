#!/usr/bin/env python3
"""Leakage check on real rows: rebuild sampled snapshots from only the messages at or before T.

    uv run python scripts/check_snapshot_leakage.py --days 2026-10-12 2026-10-14 --per-day 300

For each sampled snapshot row of the stored Stage 2 table, the state is recomputed by the plain
reference implementation from a copy of stop_updates in which every message after T was deleted,
and compared with the stored state. Any difference is a mismatch.
Writes reports/stage2_leakage.json.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.baselines.state import KEYS, STATE_COLS, UPDATE_COLS, state_reference
from rail_risk.ingest.pport_extract import train_pool_table

ROOT = Path("data/interim/pport")
SNAPS = Path("data/interim/stage2/snapshots")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=date.fromisoformat, nargs="+", required=True)
    ap.add_argument("--per-day", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    checked = mismatched = with_state = 0
    by_col = dict.fromkeys(STATE_COLS[3:], 0)
    examples = []
    for d in args.days:
        snaps = pd.read_parquet(SNAPS / f"service_date={d}" / "part.parquet")
        snaps = snaps[snaps["schedule_known"]]
        # half the sample from services that have reported something, half at random
        have = snaps[snaps["last_actual_late_s"].notna()]
        pick = pd.concat(
            [
                have.sample(min(len(have), args.per_day // 2), random_state=int(rng.integers(1e6))),
                snaps.sample(
                    min(len(snaps), args.per_day // 2), random_state=int(rng.integers(1e6))
                ),
            ]
        ).drop_duplicates(KEYS)
        updates = train_pool_table(ROOT, "stop_updates", UPDATE_COLS, [d]).to_pandas()
        updates = updates[updates["rid"].isin(set(pick["rid"]))]
        for row in pick.itertuples(index=False):
            cut = updates[(updates["rid"] == row.rid) & (updates["message_time"] <= row.snap_time)]
            one = pd.DataFrame(
                [dict(rid=row.rid, route=row.route, horizon_min=row.horizon_min,
                      snap_time=row.snap_time, dest_tpl=row.dest_tpl, pt_arr=row.pt_arr)]
            )  # fmt: skip
            ref = state_reference(cut, one).iloc[0]
            stored = pd.Series({c: getattr(row, c) for c in STATE_COLS})
            bad = [
                c
                for c in STATE_COLS[3:]
                if not (
                    (pd.isna(ref[c]) and pd.isna(stored[c]))
                    or (not pd.isna(ref[c]) and not pd.isna(stored[c]) and ref[c] == stored[c])
                )
            ]
            checked += 1
            with_state += int(not pd.isna(stored["last_actual_late_s"]))
            if bad:
                mismatched += 1
                for c in bad:
                    by_col[c] += 1
                if len(examples) < 5:
                    examples.append({"rid": row.rid, "horizon": row.horizon_min, "columns": bad})
    report = {
        "days": [str(d) for d in args.days],
        "snapshots_checked": checked,
        "snapshots_with_an_actual_by_T": with_state,
        "mismatches": mismatched,
        "mismatches_by_column": by_col,
        "examples": examples,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/stage2_leakage.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
