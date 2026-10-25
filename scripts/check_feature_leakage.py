#!/usr/bin/env python3
"""Leakage check for the Stage 3 features on real rows: rebuild each sampled row's features from
a feed cut at its snapshot time T (every table keeps only messages at or before T) and compare
with the features built from the full day. A feature that used a later message would differ.

    uv run python scripts/check_feature_leakage.py --days 2026-10-14 2026-03-10 --per-day 25
Writes reports/network_model/feature_leakage.json and exits 1 on any mismatch.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F

ROOT = Path("data/interim/pport")
SNAPS = Path("data/interim/stage2/snapshots")
FEATS = Path("data/interim/stage3/features")
COMPARE = [*F.FEATURES, "from_tpl", "from_event"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", type=date.fromisoformat, required=True)
    ap.add_argument("--per-day", type=int, default=25)
    args = ap.parse_args()
    holidays = F.bank_holidays()
    split, _ = D.load_split()
    fastest = D.ensure_fastest([*split.train, *split.validation, *split.test], list(split.train))
    checked, mismatches, with_state, details = 0, 0, 0, []
    for d in args.days:
        full = F.add_slack(pd.read_parquet(FEATS / f"service_date={d}" / "part.parquet"), fastest)
        snaps = pd.read_parquet(SNAPS / f"service_date={d}" / "part.parquet")
        pick = full.sample(min(args.per_day, len(full)), random_state=7)
        for row in pick.itertuples(index=False):
            one = snaps[
                (snaps["rid"] == row.rid)
                & (snaps["route"] == row.route)
                & (snaps["horizon_min"] == row.horizon_min)
            ]
            cut = F.build_day(ROOT, d, one, holidays, as_of=row.snap_time)
            cut = F.add_slack(cut, fastest) if not cut.empty else cut
            checked += 1
            with_state += int(row.n_actuals > 0)
            if cut.empty:
                mismatches += 1
                details.append({"day": str(d), "rid": row.rid, "problem": "no row after cut"})
                continue
            a = full[(full["rid"] == row.rid) & (full["route"] == row.route)]
            a = a[a["horizon_min"] == row.horizon_min].iloc[0]
            b = cut.iloc[0]
            bad = []
            for c in COMPARE:
                x, y = a[c], b[c]
                if pd.isna(x) or pd.isna(y):
                    same = bool(pd.isna(x) and pd.isna(y))
                elif isinstance(x, str) or isinstance(y, str):
                    same = x == y
                else:
                    same = bool(np.isclose(float(x), float(y), atol=1e-6))
                if not same:
                    bad.append({"feature": c, "full": str(x), "cut": str(y)})
            if bad:
                mismatches += 1
                details.append({"day": str(d), "rid": row.rid, "diff": bad})
    out = {
        "days": [str(d) for d in args.days],
        "rows_checked": checked,
        "rows_with_an_actual_by_T": with_state,
        "mismatches": mismatches,
        "details": details[:10],
    }
    Path("reports/network_model").mkdir(parents=True, exist_ok=True)
    Path("reports/network_model/feature_leakage.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "details"}))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
