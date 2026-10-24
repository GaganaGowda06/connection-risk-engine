#!/usr/bin/env python3
"""Compare archive actuals with HSP actuals for sampled trains on held-out routes.

HSP rows come from a scratch data directory (never data/), the archive from the extracted
stop_actuals. Stops are joined on rid and CRS (via the TIPLOC map); the difference is in minutes,
archive minus HSP, for arrivals and for departures.
    uv run python scripts/hsp_spot_check.py --hsp-dir SCRATCH --day 2026-10-13 \
        --routes lst-cbg man-shf bri-cdf
"""

import argparse
import json
import random
from datetime import date
from pathlib import Path

import pandas as pd

from rail_risk.config import load_routes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsp-dir", type=Path, required=True)
    ap.add_argument("--day", type=date.fromisoformat, required=True)
    ap.add_argument("--routes", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pport", type=Path, default=Path("data/interim/pport"))
    ap.add_argument("--stations", type=Path, default=Path("data/interim/ref/tiploc_crs.parquet"))
    args = ap.parse_args()
    routes = {r.id: r for r in load_routes("config/routes.yaml").routes}
    crs_of = pd.read_parquet(args.stations).dropna(subset=["crs"]).set_index("tiploc")["crs"]
    hsp = []
    for route in args.routes:
        for f in sorted(
            (args.hsp_dir / "interim/hsp_stops").glob(f"date={args.day}/{route}_*.parquet")
        ):
            df = pd.read_parquet(f)
            df["route"] = route
            hsp.append(df)
    hsp = pd.concat(hsp).drop_duplicates(["rid", "seq"])
    arch = pd.read_parquet(args.pport / f"stop_actuals/service_date={args.day}")
    arch["location"] = arch["tpl"].map(crs_of)
    a_arr = (
        arch[arch["event"] == "arr"]
        .dropna(subset=["location"])
        .drop_duplicates(["rid", "location"], keep="last")
    )
    a_dep = (
        arch[arch["event"] == "dep"]
        .dropna(subset=["location"])
        .drop_duplicates(["rid", "location"], keep="first")
    )
    rng = random.Random(args.seed)  # noqa: S311 - sampling, not security
    picked: list[str] = []
    per = -(-args.n // len(args.routes))
    for route in args.routes:
        dest = routes[route].destination
        rids = sorted(
            hsp[(hsp["route"] == route) & (hsp["location"] == dest) & hsp["actual_arr"].notna()][
                "rid"
            ].unique()
        )
        rng.shuffle(rids)
        picked += rids[:per]
    picked = picked[: args.n]
    sub = hsp[hsp["rid"].isin(picked)]
    arr = sub.dropna(subset=["actual_arr"]).merge(
        a_arr[["rid", "location", "actual"]], on=["rid", "location"], how="left"
    )
    dep = sub.dropna(subset=["actual_dep"]).merge(
        a_dep[["rid", "location", "actual"]], on=["rid", "location"], how="left"
    )
    out = {
        "day": str(args.day),
        "trains": len(picked),
        "per_route": {r: int(sub[sub.route == r].rid.nunique()) for r in args.routes},
    }
    for name, df, col in (("arrival", arr, "actual_arr"), ("departure", dep, "actual_dep")):
        both = df.dropna(subset=["actual"])
        diff = (both["actual"] - both[col]).dt.total_seconds() / 60
        out[name] = {
            "hsp_stops": len(df),
            "found_in_archive": len(both),
            "exact": int((diff == 0).sum()),
            "within_1_min": int((diff.abs() <= 1).sum()),
            "over_1_min": int((diff.abs() > 1).sum()),
            "mean": round(float(diff.mean()), 3),
            "max_abs": float(diff.abs().max()),
            "counts": {
                str(k): int(v) for k, v in diff.value_counts().sort_index().items() if abs(k) <= 5
            },
        }
    dest_rows = []
    for route in args.routes:
        d = routes[route].destination
        x = arr[(arr["route"] == route) & (arr["location"] == d)].dropna(subset=["actual"])
        diff = (x["actual"] - x["actual_arr"]).dt.total_seconds() / 60
        dest_rows.append(
            {
                "route": route,
                "trains": len(x),
                "exact": int((diff == 0).sum()),
                "max_abs": float(diff.abs().max()) if len(x) else None,
            }
        )
    out["destination_arrival_by_route"] = dest_rows
    missing = arr[arr["actual"].isna()]
    out["hsp_arrivals_not_in_archive_sample"] = (
        missing[["rid", "location"]].head(5).to_dict("records")
    )
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
