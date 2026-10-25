#!/usr/bin/env python3
"""Leakage test 2 for Stage 5: the online per-hub recalibration uses an outcome only after its
outbound has departed and before the scored row's T.

On real validation rows with the real model probabilities: (1) for every row the latest departure
time among the outcomes applied is before its T, and equals a brute-force recomputation on 3,000
sampled rows; (2) perturbation: flip the label of every outcome whose outbound departs at or after
a cut time C and rerun; every row scored before C must come out bit-identical, while rows scored
well after C must change (so the test can detect a leak).

    uv run python scripts/check_online_leakage.py --tag options --eta 0.0003
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rail_risk.connections import build as B
from rail_risk.connections import data as DD
from rail_risk.connections import online as O

OUT = Path("reports/connections/stage5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="options")
    ap.add_argument("--eta", type=float, default=3e-4)
    args = ap.parse_args()
    cfg = B.load_config()
    pick = json.loads((OUT / f"{args.tag}_validation.json").read_text())["summary"]["pick"]
    model = joblib.load(Path("data/interim/stage5/models") / f"{args.tag}.joblib")
    va = DD.flag_groups(DD.load_block("validation"), cfg).reset_index(drop=True)
    p = model.predict(va)[f"{pick}_cal"].to_numpy()
    t, r = O.seconds(va["snap_time"]), O.seconds(va["act_dep_out"])
    y = np.where((va["out_cancel"] == 0) & va["miss"].notna(), va["miss"], np.nan)
    hub = va["hub"].to_numpy()
    q0, used = O.run(hub, t, p, y, r, args.eta)
    after_ok = bool(np.all(~np.isfinite(used) | (used < t)))
    rng = np.random.default_rng(1)
    brute_bad = 0
    sample = rng.choice(len(va), 3000, replace=False)
    lab = np.isfinite(y) & np.isfinite(r)
    for i in sample:
        start = np.floor(t[i] / O.BLOCK_S) * O.BLOCK_S
        m = lab & (hub == hub[i]) & (r < start)
        want = r[m].max() if m.any() else np.nan
        got = used[i]
        if not ((np.isnan(want) and np.isnan(got)) or want == got):
            brute_bad += 1
    cut_ts = pd.Timestamp("2026-10-01 12:00", tz="UTC")
    C = (cut_ts - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()
    y2 = y.copy()
    flip = np.isfinite(y) & (r >= C)
    y2[flip] = 1.0 - y2[flip]
    q1, _ = O.run(hub, t, p, y2, r, args.eta)
    before = t < C
    far = t > C + 6 * 3600
    res = {
        "rows": len(va), "eta": args.eta, "option": pick,
        "latest_applied_departure_before_T_for_all_rows": after_ok,
        "brute_force_sample": 3000, "brute_force_mismatches": int(brute_bad),
        "flipped_outcomes": int(flip.sum()), "rows_before_cut": int(before.sum()),
        "rows_before_cut_changed": int((q0[before] != q1[before]).sum()),
        "rows_6h_after_cut": int(far.sum()),
        "rows_6h_after_cut_changed": int((q0[far] != q1[far]).sum()),
        "max_change_after": float(np.abs(q0[far] - q1[far]).max()),
    }  # fmt: skip
    res["passed"] = bool(
        after_ok
        and brute_bad == 0
        and res["rows_before_cut_changed"] == 0
        and res["rows_6h_after_cut_changed"] > 0
    )
    (OUT / "online_leakage.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
