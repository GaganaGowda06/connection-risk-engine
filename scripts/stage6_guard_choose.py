#!/usr/bin/env python3
# ruff: noqa: E501
"""Choose the freshness thresholds on the VALIDATION replay (scripts/stage6_guard_replay.py).

Rows are split into "intact" (T outside every removed hour and the 240 minutes after it) and
"affected". Rule, written before the run: among feed thresholds (3, 5, 8, 12, 20, 30 minutes) and
recent-silence windows (60, 120, 240) whose false-stale share of intact pair rows is at most 0.5%,
take the one with the lowest pooled pair Brier; the hub threshold is added only when it lowers the
pooled Brier with the false-stale share still at most 1%. The arrival width factor k (final width =
1 + k * hours of silence) is the value whose stale-row coverage is closest to 0.8.
Writes reports/serving/guard_threshold.json, which also opens the oot_week gate.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.serving.replay_gaps import GAPS  # noqa: F401

DB = Path("data/interim/serve/guard_replay/state/state.sqlite")
OUT = Path("reports/serving")


def load():
    con = sqlite3.connect(DB)
    arr = pd.read_sql_query("select * from arrival", con)
    pr = pd.read_sql_query("select * from pair", con)
    con.close()
    for df, cols in ((arr, ["snap_time", "pt_arr"]), (pr, ["snap_time", "plan_dep_out"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c], utc=True)
    days = sorted(arr["ssd"].unique())
    lab = pd.concat(
        [
            pd.read_parquet(
                f"data/interim/stage5/validation/service_date={d}/part.parquet",
                columns=["hub", "in_rid", "out_rid", "horizon_min", "miss", "out_cancel"],
            )
            for d in days
            if Path(f"data/interim/stage5/validation/service_date={d}").exists()
        ]
    )
    lab = lab[(lab.out_cancel == 0) & lab.miss.notna()]
    pr = pr.merge(
        lab[["hub", "in_rid", "out_rid", "horizon_min", "miss"]],
        on=["hub", "in_rid", "out_rid", "horizon_min"],
        how="inner",
    )
    snaps = pd.concat(
        [
            pd.read_parquet(
                f"data/interim/stage2/snapshots/service_date={d}/part.parquet",
                columns=["rid", "route", "horizon_min", "target_late_s"],
            )
            for d in days
            if Path(f"data/interim/stage2/snapshots/service_date={d}").exists()
        ]
    )
    arr = arr.merge(
        snaps.dropna(subset=["target_late_s"]), on=["rid", "route", "horizon_min"], how="inner"
    )
    arr["y"] = arr["target_late_s"] / 60.0
    return arr, pr


def affected(t: pd.Series) -> pd.Series:
    m = pd.Series(False, index=t.index)
    for a, b in GAPS:
        m |= (t >= a) & (t < b + pd.Timedelta(minutes=240))
    return m


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def main() -> None:
    arr, pr = load()
    pr["aff"], arr["aff"] = affected(pr["snap_time"]), affected(arr["snap_time"])
    out: dict = {
        "rows": {
            "pairs": len(pr),
            "pairs_affected": int(pr.aff.sum()),
            "arrival": len(arr),
            "arrival_affected": int(arr.aff.sum()),
        }
    }
    base = {
        "p_b_all": brier(pr.p_b, pr.miss),
        "p_b_affected": brier(pr.p_b[pr.aff], pr.miss[pr.aff]),
        "p_bar_affected": brier(pr.p_bar[pr.aff], pr.miss[pr.aff]),
        "p_b_intact": brier(pr.p_b[~pr.aff], pr.miss[~pr.aff]),
    }
    out["pair_baseline"] = base
    grid = []
    for th in (3, 5, 8, 12, 20, 30):
        for R in (60, 120, 240):
            stale = (pr.feed_age_min > th) | (pr[f"feed_gap{R}"] > th)
            false = float(stale[~pr.aff].mean())
            p = np.where(stale, pr.p_bar, pr.p_b)
            grid.append(
                {
                    "feed_min": th,
                    "recent_min": R,
                    "false_stale_intact": false,
                    "stale_share_affected": float(stale[pr.aff].mean()),
                    "brier_all": brier(p, pr.miss),
                    "brier_affected": brier(p[pr.aff], pr.miss[pr.aff]),
                    "brier_intact": brier(p[~pr.aff], pr.miss[~pr.aff]),
                }
            )
    out["pair_grid"] = grid
    ok = [g for g in grid if g["false_stale_intact"] <= 0.005]
    best = min(ok, key=lambda g: (g["brier_all"], g["false_stale_intact"]))
    th, R = best["feed_min"], best["recent_min"]
    base_stale = (pr.feed_age_min > th) | (pr[f"feed_gap{R}"] > th)
    hub_grid = []
    for hm in (10, 15, 30, 60, 1e9):
        stale = base_stale | (pr.hub_age_min > hm) | (pr.hub_gap120 > hm)
        p = np.where(stale, pr.p_bar, pr.p_b)
        hub_grid.append(
            {
                "hub_min": hm,
                "false_stale_intact": float(stale[~pr.aff].mean()),
                "brier_all": brier(p, pr.miss),
                "brier_affected": brier(p[pr.aff], pr.miss[pr.aff]),
            }
        )
    out["hub_grid"] = hub_grid
    cand = [
        g
        for g in hub_grid
        if g["hub_min"] < 1e9
        and g["false_stale_intact"] <= 0.01
        and g["brier_all"] < best["brier_all"] - 1e-5
    ]
    hub_min = min(cand, key=lambda g: g["brier_all"])["hub_min"] if cand else 1e9
    # service age, reported and not used unless it helps
    svc = []
    for sa in (30, 60, 120):
        stale = base_stale | (np.maximum(pr.in_age_min, pr.out_age_min) > sa)
        p = np.where(stale, pr.p_bar, pr.p_b)
        svc.append(
            {
                "svc_min": sa,
                "false_stale_intact": float(stale[~pr.aff].mean()),
                "brier_all": brier(p, pr.miss),
                "brier_affected": brier(p[pr.aff], pr.miss[pr.aff]),
            }
        )
    out["service_age_grid_reported_not_used"] = svc
    # arrival
    stale_a = (arr.feed_age_min > th) | (arr[f"feed_gap{R}"] > th)
    out["arrival_stale_share"] = {
        "intact": float(stale_a[~arr.aff].mean()),
        "affected": float(stale_a[arr.aff].mean()),
    }
    cov = []
    hrs = (np.maximum(arr.feed_age_min, arr[f"feed_gap{R}"]) / 60.0).clip(lower=0)
    for k in (0.0, 0.5, 1.0, 2.0, 4.0):
        f = 1.0 + k * hrs
        lo = arr.point - f * (arr.point - arr.static_lo)
        hi = arr.point + f * (arr.static_hi - arr.point)
        m = stale_a & arr.aff
        cov.append(
            {
                "k": k,
                "coverage_stale_rows": float(((lo <= arr.y) & (arr.y <= hi))[m].mean())
                if m.any()
                else None,
                "n": int(m.sum()),
                "mean_width": float((hi - lo)[m].mean()) if m.any() else None,
            }
        )
    out["arrival_widen_grid"] = cov
    out["arrival_aci_coverage_stale_rows"] = (
        float(((arr.aci_lo <= arr.y) & (arr.y <= arr.aci_hi))[stale_a & arr.aff].mean())
        if (stale_a & arr.aff).any()
        else None
    )
    usable = [c for c in cov if c["coverage_stale_rows"] is not None and c["n"] > 20]
    k = min(usable, key=lambda c: abs(c["coverage_stale_rows"] - 0.8))["k"] if usable else 1.0
    choice = {
        "feed_min": th,
        "recent_min": R,
        "hub_min": hub_min,
        "widen_k": k,
        "chosen_on": "validation replay 2026-10-01 to 2026-10-04 with removed hours 2026-10-02 09:00-10:00 and 2026-10-03 15:00-17:00 UTC",
        "pair_brier_affected": {
            "without_guard": base["p_b_affected"],
            "with_guard": best["brier_affected"],
            "gap_bucket": base["p_bar_affected"],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "guard_choice.json").write_text(json.dumps(out, indent=1, default=float))
    (OUT / "guard_threshold.json").write_text(json.dumps(choice, indent=1, default=float))
    print(json.dumps(choice, indent=1, default=float))
    print(json.dumps(base))
    print(
        [
            (
                g["feed_min"],
                g["recent_min"],
                round(g["false_stale_intact"], 4),
                round(g["brier_affected"], 4),
            )
            for g in grid
        ]
    )


if __name__ == "__main__":
    main()
