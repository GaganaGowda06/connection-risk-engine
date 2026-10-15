"""Latency of a running risk API. Not part of CI: timings depend on the machine.

usage: uv run python scripts/bench_api.py [--url http://127.0.0.1:8000] [--n 200] [--rid RID]

Sends sequential POST /risk requests for one service (the soonest due, unless --rid is given)
and reports p50, p95 and max in milliseconds. A 202 warming answer is retried until it clears
and is not counted in the timings.
"""

import argparse
import statistics
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rid", default=None)
    args = ap.parse_args()
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        rid = args.rid
        if rid is None:
            services = client.get("/services").json()
            if not services:
                print("no services due in the next three hours", file=sys.stderr)
                return 1
            rid = services[0]["rid"]
        for _ in range(10):  # let a cache miss warm
            if client.post("/risk", json={"rid": rid, "fare": 50}).status_code != 202:
                break
            time.sleep(2)
        times, codes = [], {}
        for i in range(args.n):
            started = time.perf_counter()
            r = client.post("/risk", json={"rid": rid, "fare": float(i % 50)})
            times.append((time.perf_counter() - started) * 1000)
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
    times.sort()
    p95 = times[max(0, int(0.95 * len(times)) - 1)]
    print(
        f"{args.n} requests for {rid}: p50 {statistics.median(times):.1f} ms, p95 {p95:.1f} ms, "
        f"max {times[-1]:.1f} ms, status counts {codes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
