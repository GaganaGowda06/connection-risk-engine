#!/usr/bin/env python3
# ruff: noqa: E501
"""Load test of the running API at a stated, fixed request rate (open loop, so a slow server cannot
slow the offered rate down). Mix: 50% /risk, 35% /connection, 10% /inbound, 5% /health, drawn from
real keys in the committed ledger. Reports p50, p95, p99 and errors per endpoint and overall.

    uv run python scripts/stage6_loadtest.py --state data/interim/serve/state_oot --rate 50 --seconds 60
"""

import argparse
import asyncio
import json
import sqlite3
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


def pct(a) -> dict:
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)), "p50": round(float(np.percentile(a, 50)), 1),
        "p95": round(float(np.percentile(a, 95)), 1), "p99": round(float(np.percentile(a, 99)), 1),
        "max": round(float(a.max()), 1),
    }  # fmt: skip


async def run(base: str, keys: dict, rate: float, seconds: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    keys = {k: np.asarray(v) for k, v in keys.items()}
    lat: dict[str, list] = {"risk": [], "connection": [], "inbound": [], "health": []}
    errors: dict[str, int] = {k: 0 for k in lat}
    status: dict[str, dict] = {k: {} for k in lat}
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
    async with httpx.AsyncClient(base_url=base, timeout=10.0, limits=limits) as client:

        async def one(kind: str):
            t = time.perf_counter()
            try:
                if kind == "risk":
                    r = await client.post("/risk", json={"rid": str(rng.choice(keys["rids"]))})
                elif kind == "connection":
                    a, b = (str(x) for x in keys["pairs"][rng.integers(len(keys["pairs"]))])
                    r = await client.post("/connection", json={"in_rid": a, "out_rid": b})
                elif kind == "inbound":
                    r = await client.get(
                        "/inbound",
                        params={
                            "hub": str(rng.choice(["BHM", "MAN", "BRI"])),
                            "at": str(rng.choice(keys["inbound_at"])),
                        },
                    )
                else:
                    r = await client.get("/health")
                ms = (time.perf_counter() - t) * 1000
                lat[kind].append(ms)
                status[kind][r.status_code] = status[kind].get(r.status_code, 0) + 1
                if r.status_code >= 500:
                    errors[kind] += 1
            except Exception:
                errors[kind] += 1

        kinds = rng.choice(
            ["risk", "connection", "inbound", "health"],
            size=int(rate * seconds),
            p=[0.5, 0.35, 0.1, 0.05],
        )
        tasks = []
        t0 = time.perf_counter()
        for i, k in enumerate(kinds):
            delay = i / rate - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(str(k))))
        await asyncio.gather(*tasks)
    return {
        "lat": lat,
        "errors": errors,
        "status": status,
        "wall": time.perf_counter() - t0,
        "n": len(kinds),
    }


def _proc(args):
    base, keys, rate, seconds, seed = args
    return asyncio.run(run(base, keys, rate, seconds, seed))


def summarise(parts: list, rate: float, seconds: float, procs: int) -> dict:
    lat = {k: [x for p in parts for x in p["lat"][k]] for k in parts[0]["lat"]}
    errors = {k: sum(p["errors"][k] for p in parts) for k in lat}
    status: dict = {k: {} for k in lat}
    for p in parts:
        for k, d in p["status"].items():
            for c, n in d.items():
                status[k][c] = status[k].get(c, 0) + n
    allv = np.array([x for v in lat.values() for x in v])
    return {"rate_per_s": rate, "generator_processes": procs, "seconds": seconds,
            "requests": int(sum(p["n"] for p in parts)), "wall_s": round(max(p["wall"] for p in parts), 1),
            "overall_ms": pct(allv), "by_endpoint_ms": {k: pct(np.array(v)) for k, v in lat.items()},
            "errors_5xx_or_failed": errors, "status_codes": status}  # fmt: skip


def rids_days(state: str) -> list:
    con = sqlite3.connect(Path(state) / "state.sqlite")
    rows = [r[0] for r in con.execute("select distinct ssd from arrival")]
    con.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--rate", type=float, default=50)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--procs", type=int, default=1, help="generator processes (rate is split)")
    ap.add_argument("--out", default="reports/serving/loadtest.json")
    args = ap.parse_args()
    con = sqlite3.connect(Path(args.state) / "state.sqlite")
    rids = pd.read_sql_query("select distinct rid from arrival limit 5000", con)["rid"].tolist()
    pairs = (
        pd.read_sql_query(
            "select in_rid, out_rid from pair group by in_rid, out_rid limit 5000", con
        )
        .to_numpy()
        .tolist()
    )
    con.close()
    # busy replay clocks (06:00 to 21:00 UTC on the ledger's days), so /inbound returns real rows
    days = sorted({str(x)[:10] for x in rids_days(args.state)})
    at = [f"{d} {h:02d}:{m:02d}" for d in days for h in range(6, 21) for m in (0, 30)]
    keys = {"rids": rids, "pairs": pairs, "inbound_at": at}
    jobs = [(args.base, keys, args.rate / args.procs, args.seconds, i) for i in range(args.procs)]
    if args.procs == 1:
        parts = [_proc(jobs[0])]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.procs) as pool:
            parts = pool.map(_proc, jobs)
    res = summarise(parts, args.rate, args.seconds, args.procs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
