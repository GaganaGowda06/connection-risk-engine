#!/usr/bin/env python3
"""Compare DuckDB, Polars and Spark on building snapshot state for one week of stop_updates.

    uv run python scripts/bench_state_engines.py --start 2026-10-10 --end 2026-10-16

The parent runs each engine in its own process (`--engine NAME`), samples the resident memory of
the whole process tree (the Spark JVM included) and then checks the outputs are identical.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import psutil

OUT = Path("data/interim/stage2/engine_bench")
ROOT = Path("data/interim/pport")


def child(engine: str, start: date, end: date) -> None:
    from rail_risk.baselines.services import day_services, snapshot_rows
    from rail_risk.baselines.state import ENGINES, UPDATE_COLS
    from rail_risk.clean.run import route_specs
    from rail_risk.config import load_routes
    from rail_risk.ingest.pport_extract import train_pool_table

    stations = pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
    specs = route_specs(load_routes("config/routes.yaml"), stations)
    parts, rows_in = [], 0
    for i in range((end - start).days + 1):
        d = start + timedelta(days=i)
        snaps = snapshot_rows(day_services(ROOT, d, specs))
        snaps = snaps[snaps["schedule_known"]]
        updates = train_pool_table(ROOT, "stop_updates", UPDATE_COLS, [d])
        rows_in += updates.num_rows
        parts.append(ENGINES[engine](updates, snaps))
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT / f"{engine}.parquet")
    print(json.dumps({"rows_in": rows_in, "snapshots": len(out)}))


def run_engine(engine: str, start: date, end: date) -> dict:
    cmd = [sys.executable, __file__, "--engine", engine, "--start", str(start), "--end", str(end)]
    t0 = time.monotonic()
    proc = subprocess.Popen(  # noqa: S603
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    peak = 0
    stop = threading.Event()

    def poll() -> None:
        nonlocal peak
        p = psutil.Process(proc.pid)
        while not stop.is_set():
            try:
                procs = [p, *p.children(recursive=True)]
                peak = max(peak, sum(q.memory_info().rss for q in procs if q.is_running()))
            except psutil.Error:
                pass
            time.sleep(0.25)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    out, _ = proc.communicate()
    stop.set()
    th.join()
    info = json.loads(out.strip().splitlines()[-1]) if proc.returncode == 0 and out.strip() else {}
    return {
        "engine": engine,
        "seconds": round(time.monotonic() - t0, 1),
        "peak_gb": round(peak / 1e9, 2),
        "returncode": proc.returncode,
        **info,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine")
    ap.add_argument("--engines", nargs="+", default=["duckdb", "polars", "spark"])
    ap.add_argument("--tag", default="week")
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    args = ap.parse_args()
    if args.engine:
        child(args.engine, args.start, args.end)
        return
    results = [run_engine(e, args.start, args.end) for e in args.engines]
    frames = {
        r["engine"]: pd.read_parquet(OUT / f"{r['engine']}.parquet")
        for r in results
        if r["returncode"] == 0
    }
    base = frames.get(args.engines[0])
    for r in results:
        f = frames.get(r["engine"])
        r["identical_to_first"] = bool(f is not None and base is not None and f.equals(base))
        if f is not None and base is not None and not r["identical_to_first"]:
            same_shape = f.shape == base.shape
            r["differing_rows"] = (
                int((f.fillna(-1) != base.fillna(-1)).any(axis=1).sum()) if same_shape else "shape"
            )
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/stage2_engine_bench_{args.tag}.json").write_text(
        json.dumps(results, indent=1) + "\n"
    )
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
