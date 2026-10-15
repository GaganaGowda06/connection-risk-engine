"""Feature parity and actual-time agreement between HSP and Darwin for one service date.

usage: uv run python scripts/live_parity.py 2026-10-15 [--from 11:45] [--n 20]

Needs Darwin raw files and HSP stops for that date (HSP is ingested with ingest-hsp). Services
are bhm-eus services present in both feeds whose scheduled arrival is after --from (UTC), so
Darwin had been capturing for at least the hour before the forecast time.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from rail_risk.config import load_routes
from rail_risk.live import (
    STOP_COLUMNS,
    FeatureService,
    LiveStore,
    actual_time_agreement,
    load_tiplocs,
    read_parts,
    resolve_route,
    source_parity,
)
from rail_risk.spark import build_spark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--from", dest="after", default="11:45")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--minutes", type=int, default=60)
    args = ap.parse_args()
    interim = Path("data/interim")
    store = LiveStore(interim, Path("data/raw/darwin"), load_tiplocs())
    store.refresh()
    hsp = read_parts(interim / "hsp_stops" / f"date={args.date}", STOP_COLUMNS)
    routes = load_routes("config/routes.yaml").routes
    features = FeatureService(
        build_spark("parity", cores=2, driver_memory="2g", shuffle_partitions=2, adaptive=False),
        interim,
        Path("config/routes.yaml"),
    )
    floor = pd.Timestamp(f"{args.date} {args.after}", tz="UTC")
    pairs = {}
    for rid, h in hsp.groupby("rid"):
        h = h.sort_values("seq").reset_index(drop=True)
        dest = h[h["location"] == "EUS"]
        if dest.empty or pd.isna(dest["sched_arr"].iloc[0]) or pd.isna(dest["actual_arr"].iloc[0]):
            continue
        if dest["sched_arr"].iloc[0] < floor or not store.has(rid):
            continue
        d = store.stop_table(rid)
        if len(d) >= 3 and resolve_route(d, routes) == "bhm-eus":
            pairs[rid] = (h.assign(date_of_service=pd.to_datetime(h["date_of_service"])), d)
    random.seed(7)
    sample = {r: pairs[r] for r in random.sample(sorted(pairs), min(args.n, len(pairs)))}
    out = {
        "date": args.date,
        "services_in_both_feeds": len(pairs),
        "sample": source_parity(features, sample, "bhm-eus", args.minutes),
        "all": source_parity(features, pairs, "bhm-eus", args.minutes),
        "actual_times_sample": actual_time_agreement(sample),
        "actual_times_all": actual_time_agreement(pairs),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
