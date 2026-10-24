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
    route.groupby(["route", "month"], as_index=False)[
        ["scheduled", "ran", "cancelled"]
    ].sum().to_csv(OUT / "stage1_coverage_routes.csv", index=False)
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
