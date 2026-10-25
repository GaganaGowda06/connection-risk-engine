#!/usr/bin/env python3
"""Build the Stage 5 pair tables, one parquet per service date, from train_pool only.

    uv run python scripts/stage5_build.py --blocks train validation test   (February: --february)
Train and validation pairs are thinned (config thin); test and February keep every pair.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from rail_risk.baselines.split import network_split
from rail_risk.connections import build as B
from rail_risk.ingest.pport_extract import reading_split

DATA = Path("data/interim/stage5")


def _one(args):
    d, block, thin, feb = args
    cfg = B.load_config()
    mct = B.load_mct_table()
    tc = pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
    dest = DATA / block / f"service_date={d}" / "part.parquet"
    if dest.exists():
        return {"ssd": str(d), "skipped": True}
    t0 = time.time()
    if feb:
        with reading_split("winter_test", "february_stress"):
            df, st = B.build_day(Path("data/interim/pport"), d, cfg, mct, tc, thin)
    else:
        df, st = B.build_day(Path("data/interim/pport"), d, cfg, mct, tc, thin)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    st["seconds"] = round(time.time() - t0, 1)
    st["peak_rss_mb"] = B._rss_mb()
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=[])
    ap.add_argument("--february", action="store_true")
    ap.add_argument("--days", nargs="*", help="limit to these dates (smoke test)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    cfg = B.load_config()
    split, _ = network_split()
    jobs = []
    for block in args.blocks:
        days = getattr(split, {"train": "train", "validation": "validation", "test": "test"}[block])
        thin = cfg["thin"] if block == "train" else 1
        jobs += [(d, block, thin, False) for d in days]
    if args.february:
        from rail_risk.connections.build import february_days

        jobs += [(d, "february", 1, True) for d in february_days()]
    if args.days:
        keep = {date.fromisoformat(x) for x in args.days}
        jobs = [j for j in jobs if j[0] in keep]
    out = []
    if args.workers == 1:  # in process, one day at a time
        for j in jobs:
            st = _one(j)
            if st.get("peak_rss_mb", 0) > 4096:
                raise SystemExit(
                    f"stopped: peak memory {st['peak_rss_mb']} MB over 4 GB on {st['ssd']}"
                )
            out.append(st)
            print(json.dumps(st), flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            for st in ex.map(_one, jobs):
                out.append(st)
                print(json.dumps(st), flush=True)
    (DATA / "build_stats.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with (DATA / "build_stats.jsonl").open("a") as fh:
        for st in out:
            fh.write(json.dumps(st) + "\n")


if __name__ == "__main__":
    main()
