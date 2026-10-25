#!/usr/bin/env python3
"""Leakage check for the adaptive ranges on real rows: for sampled validation held-out rows, rerun
the stream with only the outcomes that had arrived by the row's snapshot time T (plus the row), and
compare the row's range with the full run. Any difference means a later outcome was used.

    uv run python scripts/check_adaptive_leakage.py --run-id MODELRUN [--rows 120]
Writes reports/network_model/stage4b/adaptive_leakage.json; exits 1 on any mismatch."""

import argparse
import json

import numpy as np
import pandas as pd

from rail_risk.network_model import adaptive as A
from rail_risk.network_model import stage4b as B


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rows", type=int, default=120)
    args = ap.parse_args()
    choice = B.load_choice()["choice"]
    cal, rows, held, fit, gs = B.load_context(args.run_id)
    val = rows["validation"]
    hv = B.with_predictions(cal, fit, val[val["route"].isin(held)].reset_index(drop=True))
    out = {"method": choice["method"], "gamma": choice["gamma"], "rows": 0, "mismatches": 0}
    rng = np.random.default_rng(3)
    for method, gamma in (
        (choice["method"], choice["gamma"] or 0.0),
        ("rolling", 0.0),
        ("aci_group", 0.01),
    ):
        full = A.run(hv, gs, method, gamma)
        arr = hv["pt_arr"] + pd.to_timedelta(hv["target_min"], unit="m")
        for i in rng.choice(len(hv), args.rows, replace=False):
            if method == "aci_group":
                same = np.ones(len(hv), bool)
            else:
                same = (
                    (hv["route"] == hv.loc[i, "route"])
                    & (hv["horizon_min"] == hv.loc[i, "horizon_min"])
                ).to_numpy()
            known = same & (arr <= hv.loc[i, "snap_time"]).to_numpy()
            known[i] = True
            sub = hv[known]
            res = A.run(sub, gs, method, gamma)
            out["rows"] += 1
            if not (
                res.loc[i, "lo"] == full.loc[i, "lo"] and res.loc[i, "hi"] == full.loc[i, "hi"]
            ):
                out["mismatches"] += 1
    (B.OUT / "adaptive_leakage.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return 1 if out["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
