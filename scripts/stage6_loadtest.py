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


async def run(base: str, keys: dict, rate: float, seconds: float) -> dict:
    rng = np.random.default_rng(0)
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
                    a, b = keys["pairs"][rng.integers(len(keys["pairs"]))]
                    r = await client.post("/connection", json={"in_rid": a, "out_rid": b})
                elif kind == "inbound":
                    r = await client.get(
                        "/inbound", params={"hub": str(rng.choice(["BHM", "MAN", "BRI"]))}
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
        wall = time.perf_counter() - t0
    allv = np.array([x for v in lat.values() for x in v])
    return {"rate_per_s": rate, "seconds": seconds, "requests": int(len(kinds)), "wall_s": round(wall, 1), "overall_ms": pct(allv),
            "by_endpoint_ms": {k: pct(np.array(v)) for k, v in lat.items()}, "errors_5xx_or_failed": errors, "status_codes": status}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--rate", type=float, default=50)
    ap.add_argument("--seconds", type=float, default=60)
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
    res = asyncio.run(run(args.base, {"rids": rids, "pairs": pairs}, args.rate, args.seconds))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
