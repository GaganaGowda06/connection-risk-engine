#!/usr/bin/env python3
"""Run the Stage 1 checks and the coverage counts over every extracted service date.

Writes reports/stage1_cleaning.csv (check, route, rows, flagged), stage1_coverage_routes.csv,
stage1_coverage_operators.csv and stage1_schedule_share.csv. Routes come from config/routes.yaml.
"""

import argparse
import time
from datetime import date
from pathlib import Path

import pandas as pd

from rail_risk.clean.run import process_day, route_specs, summarise
from rail_risk.config import load_routes

OUT = Path("reports")


def write_coverage(by_toc: pd.DataFrame, routes) -> None:
    """Test 1 evidence: per route and month, trains of the table's operator and of any operator."""
    months = sorted(by_toc["month"].unique())
    rows = []
    for r in routes.routes:
        if r.legacy:
            continue
        mine = by_toc[by_toc["route"] == r.id]
        for m in months:
            in_month = mine[mine["month"] == m]
            own = in_month[in_month["toc"] == r.toc]
            rows.append(
                {
                    "route": r.id,
                    "hub": r.hub,
                    "operator": r.operator,
                    "toc": r.toc,
                    "length": r.length,
                    "set": "held_out" if r.held_out else "train",
                    "month": m,
                    "scheduled_operator": int(own["scheduled"].sum()),
                    "ran_operator": int(own["ran"].sum()),
                    "cancelled_operator": int(own["cancelled"].sum()),
                    "scheduled_all_operators": int(in_month["scheduled"].sum()),
                }
            )
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / "stage1_coverage.csv", index=False)
    empty = cov[cov["scheduled_operator"] == 0]
    print(f"routes {cov['route'].nunique()} months {len(months)} empty route-months {len(empty)}")
    if len(empty):
        print(empty[["route", "month"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/interim/pport"))
    ap.add_argument("--stations", type=Path, default=Path("data/interim/ref/tiploc_crs.parquet"))
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    args = ap.parse_args()
    stations = pd.read_parquet(args.stations)
    specs = route_specs(load_routes("config/routes.yaml"), stations)
    days = sorted(
        date.fromisoformat(p.name.split("=")[1])
        for p in (args.root / "stop_actuals").glob("service_date=*")
    )
    days = [
        d for d in days if (not args.start or d >= args.start) and (not args.end or d <= args.end)
    ]
    per_day = []
    started = time.monotonic()
    for i, d in enumerate(days):
        per_day.append(process_day(args.root, d, specs, stations))
        if i % 10 == 0:
            print(f"{d} {i + 1}/{len(days)} {time.monotonic() - started:.0f}s", flush=True)
    OUT.mkdir(exist_ok=True)
    summarise(per_day).to_csv(OUT / "stage1_cleaning.csv", index=False)
    by_day = pd.DataFrame([r for d in per_day for r in d["checks"]])
    by_day[by_day["route"] != "network"].to_csv(OUT / "stage1_cleaning_by_day.csv", index=False)
    route = pd.DataFrame([r for d in per_day for r in d["route_cov"]])
    by_toc = route.groupby(["route", "toc", "month"], as_index=False)[
        ["scheduled", "ran", "cancelled"]
    ].sum()
    by_toc.to_csv(OUT / "stage1_coverage_routes.csv", index=False)
    write_coverage(by_toc, load_routes("config/routes.yaml"))
    toc = pd.DataFrame([r for d in per_day for r in d["toc_cov"]])
    toc.groupby(["toc", "month"], as_index=False)["scheduled"].sum().to_csv(
        OUT / "stage1_coverage_operators.csv", index=False
    )
    pd.DataFrame([d["share"] for d in per_day]).to_csv(
        OUT / "stage1_schedule_share.csv", index=False
    )
    print(f"done {len(days)} days in {time.monotonic() - started:.0f}s")


if __name__ == "__main__":
    main()
