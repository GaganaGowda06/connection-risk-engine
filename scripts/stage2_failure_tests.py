#!/usr/bin/env python3
"""Break the Stage 2 baselines on real data: a missing day, an empty route and the split guard.

    uv run python scripts/stage2_failure_tests.py

Nothing under data/ is changed: the missing-day and empty-route cases drop rows from copies in
memory or in a scratch directory. Writes reports/stage2_failure_tests.json.
"""

import json
import shutil
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from rail_risk.baselines import build, models, scoring
from rail_risk.baselines.run import SNAP_DIR
from rail_risk.baselines.split import network_split
from rail_risk.clean.run import route_specs
from rail_risk.config import load_routes
from rail_risk.ingest.darwin_pport import load_stage1_config, split_for

ROOT = Path("data/interim/pport")


def score(train, test, held, routes):
    f = models.fit(train)
    stats = {m: scoring.row_stats(test, p) for m, p in models.predict(f, test).items()}
    return scoring.score_all(stats, held, routes, 200, 1, 0.95)


def main() -> None:
    split, _ = network_split()
    routes = [r for r in load_routes("config/routes.yaml").routes if not r.legacy]
    held = {r.id for r in routes if r.held_out}
    ids = [r.id for r in routes]
    train = build.load_snapshots(SNAP_DIR, list(split.train))
    test = build.load_snapshots(SNAP_DIR, list(split.test))
    base = score(train, test, held, ids)

    def pick(df):
        mask = (df.method == "persistence") & (df.group == "held_out") & (df.horizon_min == 15)
        return df[mask].iloc[0]

    out: dict = {
        "baseline_held_out_persistence_15": {
            "n": int(pick(base)["n"]),
            "n_dates": int(pick(base)["n_dates"]),
            "mae": round(float(pick(base)["mae"]), 3),
        }
    }

    # 1. a day with no extract: rebuild the range into a scratch dir including days that have no
    # train_pool data (the out-of-time week, and Christmas Day with no service)
    cfg = load_stage1_config()
    stations = pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
    specs = route_specs(type(load_routes("config/routes.yaml"))(routes=routes), stations)
    with tempfile.TemporaryDirectory() as tmp:
        probe_days = [date(2026, 10, 17), date(2025, 12, 25), date(2026, 10, 14)]
        rep = build.build_range(ROOT, probe_days, specs, Path(tmp))
        out["missing_day_build"] = {
            "days": [str(d) for d in probe_days],
            "split_of_each": [split_for(d, cfg) for d in probe_days],
            "built": rep["built"],
            "missing_days": rep["missing_days"],
        }
    # 2. a day dropped from the test block: scoring continues on the remaining dates
    dropped = test[test["ssd"] != date(2026, 10, 13)]
    r = score(train, dropped, held, ids)
    out["test_block_missing_2026-10-13"] = {
        "n_dates": int(pick(r)["n_dates"]),
        "n": int(pick(r)["n"]),
        "mae": round(float(pick(r)["mae"]), 3),
    }
    # 3. an empty route: all of its test rows dropped
    gone = test[test["route"] != "mcv-liv"]
    r = score(train, gone, held, ids)
    row = r[(r.method == "historical") & (r.group == "mcv-liv") & (r.horizon_min == 15)].iloc[0]
    out["empty_route_mcv-liv"] = {
        "n": int(row["n"]),
        "metrics_reported": bool("mae" in row and pd.notna(row.get("mae"))),
        "other_routes_still_scored": int(
            (
                r[(r.method == "historical") & (r.horizon_min == 15) & (r.group.isin(ids))]["n"] > 0
            ).sum()
        ),
    }
    # 4. the split guard: no oot_week or winter_test date reached any fit or snapshot
    dirs = sorted(p.name.split("=")[1] for p in SNAP_DIR.glob("service_date=*"))
    labels = pd.Series([split_for(date.fromisoformat(d), cfg) for d in dirs]).value_counts()
    out["snapshot_dates_by_split"] = labels.to_dict()
    out["oot_week_or_winter_test_dates_in_snapshots"] = int(
        labels.get("oot_week", 0) + labels.get("winter_test", 0)
    )
    out["fit_dates_outside_train_pool"] = sum(
        split_for(d, cfg) != "train_pool" for d in [*split.train, *split.validation, *split.test]
    )
    shutil.rmtree(Path("data/interim/stage2/scratch"), ignore_errors=True)
    Path("reports/stage2_failure_tests.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
