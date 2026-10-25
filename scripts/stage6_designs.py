#!/usr/bin/env python3
# ruff: noqa: E501
"""Compare three state-store designs on a one-hour replay of a validation hour (train_pool only).

Each design runs in its own fresh process: preload what was received before the hour, process the
hour, answer 1,000 /risk and 1,000 /connection lookups, then restart (a new engine over the saved
state) and time it. Reports latency, peak memory (resident set), state size and restart time.

    uv run python scripts/stage6_designs.py
"""

import json
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reports/serving")
HOUR = pd.Timestamp("2026-09-25 12:00", tz="UTC")
SSDS = [date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 26)]


def one(kind: str) -> dict:
    import resource

    from rail_risk.serving import engine as E
    from rail_risk.serving import state as S

    work = Path(f"data/interim/serve/design_{kind}")
    if work.exists():
        shutil.rmtree(work)
    store = S.open_store(kind, work / "state")
    eng = E.Engine(store, Path("data/interim/pport"), work / "store", ssds=SSDS)
    eng.preload(HOUR, SSDS)
    t0 = time.perf_counter()
    stats = eng.process_hour(HOUR)
    process_s = time.perf_counter() - t0
    if kind == "memory":
        t = time.perf_counter()
        store.checkpoint()
        stats["seconds"]["checkpoint"] = round(time.perf_counter() - t, 3)
    arr, pairs = store.all("arrival"), store.all("pair")
    rng = np.random.default_rng(0)
    rids = rng.choice(arr["rid"].to_numpy(), 1000)
    pk = pairs[["in_rid", "out_rid"]].sample(1000, random_state=0, replace=True).to_numpy()
    lat_r, lat_c = [], []
    for r in rids:
        t = time.perf_counter()
        store.risk(r)
        lat_r.append((time.perf_counter() - t) * 1000)
    for a, b in pk:
        t = time.perf_counter()
        store.pair(a, b)
        lat_c.append((time.perf_counter() - t) * 1000)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    store.close()
    size = sum(f.stat().st_size for f in (work / "state").rglob("*") if f.is_file())
    # restart: a new engine over the saved state
    t = time.perf_counter()
    store2 = S.open_store(kind, work / "state")
    E.Engine(store2, Path("data/interim/pport"), work / "store", ssds=SSDS)
    restart_s = time.perf_counter() - t
    return {
        "design": kind, "process_hour_s": round(process_s, 2), "hour_stats": stats,
        "risk_lookup_ms": {"p50": round(float(np.percentile(lat_r, 50)), 2), "p95": round(float(np.percentile(lat_r, 95)), 2)},
        "connection_lookup_ms": {"p50": round(float(np.percentile(lat_c, 50)), 2), "p95": round(float(np.percentile(lat_c, 95)), 2)},
        "peak_mb": round(peak), "state_mb": round(size / 1e6, 1), "restart_s": round(restart_s, 2),
        "rows": {"arrival": len(arr), "pair": len(pairs)},
    }  # fmt: skip


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        print("RESULT " + json.dumps(one(sys.argv[2])))
        return
    res = []
    for kind in ("memory", "duckdb", "sqlite"):
        p = subprocess.run(  # noqa: S603
            [sys.executable, __file__, "--one", kind], capture_output=True, text=True, check=False
        )
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            raise SystemExit(f"{kind} failed:\n{p.stderr[-1500:]}")
        res.append(json.loads(line[0][7:]))
        print(
            json.dumps(
                {
                    k: res[-1][k]
                    for k in (
                        "design",
                        "process_hour_s",
                        "peak_mb",
                        "state_mb",
                        "restart_s",
                        "risk_lookup_ms",
                        "connection_lookup_ms",
                    )
                }
            ),
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design_comparison.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
