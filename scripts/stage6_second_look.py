#!/usr/bin/env python3
# ruff: noqa: E501
"""SECOND LOOK at the 2026-10-20 feed gap on oot_week, once, after the guard was frozen elsewhere.

oot_week is closed for tuning: the frozen guard (reports/serving/guard_v2.json, fitted on simulated
gaps from train and validation days) is applied to the ledger the Stage 6 replay already wrote, and
only the outcomes of service date 2026-10-20 are read, through the guarded reader with the purpose
oot_gap_second_look. The window is the one the first look used (snapshots 09:00 to 15:00 UTC), so
the numbers are comparable. Compared: the model as served by no guard, the first Stage 6 guard
(binary, silence over 12 minutes in the last 60), and the frozen guard. Writes
reports/serving/oot_gap_second_look.json, which is also the marker that closes this reader.

    uv run python scripts/stage6_second_look.py --state data/interim/serve/state_oot
"""

import argparse
import json
import sqlite3
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.baselines import build as BB
from rail_risk.connections import build as CB
from rail_risk.ingest.pport_extract import GATE_SECOND, reading_split
from rail_risk.serving import artifacts as AR
from rail_risk.serving.guard import Guard

OUT = Path("reports/serving")
DAY = date(2026, 10, 20)
LO, HI = pd.Timestamp("2026-10-20 09:00", tz="UTC"), pd.Timestamp("2026-10-20 15:00", tz="UTC")


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="data/interim/serve/state_oot")
    args = ap.parse_args()
    dirty = subprocess.run(  # noqa: S603
        ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit("run the second look from a clean commit")
    if GATE_SECOND.exists():
        raise SystemExit(f"the second look was already done: {GATE_SECOND}")
    guard = Guard.from_json(json.loads((OUT / "guard_v2.json").read_text()))
    v1 = Guard(design="binary", bin_min=12, bin_window=60, flag_min=12, flag_window=60)
    none = Guard(design="none")
    con = sqlite3.connect(Path(args.state) / "state.sqlite")
    arr = pd.read_sql_query("select * from arrival where ssd = ?", con, params=(str(DAY),))
    pr = pd.read_sql_query("select * from pair where ssd = ?", con, params=(str(DAY),))
    con.close()
    for df, cols in ((arr, ["snap_time", "pt_arr"]), (pr, ["snap_time"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c], utc=True)
    arr = arr[(arr.snap_time >= LO) & (arr.snap_time < HI)].reset_index(drop=True)
    pr = pr[(pr.snap_time >= LO) & (pr.snap_time < HI)].reset_index(drop=True)
    for c in ("nya", "held_group"):
        pr[c] = pr[c].astype(bool)
    root = Path("data/interim/pport")
    cfg = CB.load_config()
    mct, tc = CB.load_mct_table(), pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
    with reading_split("oot_week", "oot_gap_second_look"):
        t = BB.day_targets(root, DAY).rename(columns={"tpl": "dest_tpl", "actual": "actual_arr"})
        lab, _ = CB.build_day(root, DAY, cfg, mct, tc, 1)
    a = arr.merge(t.assign(ssd=str(DAY)), on=["ssd", "rid", "dest_tpl"], how="left")
    a["gap"] = (a["actual_arr"] - a["pt_arr"]).abs()
    a = a.sort_values(["rid", "route", "horizon_min", "gap"], na_position="last").drop_duplicates(
        ["rid", "route", "horizon_min"]
    )
    a.loc[a["gap"] > pd.Timedelta(hours=12), "actual_arr"] = pd.NaT
    a["y"] = (a["actual_arr"] - a["pt_arr"]).dt.total_seconds() / 60.0
    a = a[a.y.notna()].reset_index(drop=True)
    lab["ssd"] = lab["ssd"].astype(str)
    p = pr.merge(
        lab[["ssd", "hub", "in_rid", "out_rid", "horizon_min", "miss", "out_cancel", "act_arr_in"]],
        on=["ssd", "hub", "in_rid", "out_rid", "horizon_min"],
        how="inner",
    )
    p = p[(p.out_cancel == 0) & p.miss.notna()].reset_index(drop=True)
    held = set(AR.load()["held"])
    h = a[a.route.isin(held)].reset_index(drop=True)
    res = {"label": "SECOND LOOK at the 2026-10-20 window (oot_week was closed for tuning; the guard was fixed on simulated gaps first)",
           "window_utc": [str(LO), str(HI)], "pair_rows": len(p), "arrival_rows": len(a), "arrival_held_out_rows": len(h)}  # fmt: skip
    out = {}
    for name, g in (("no_guard", none), ("stage6_first_guard", v1), ("frozen_guard", guard)):
        gp, ga = g.pairs(p), g.arrivals(h)
        o15 = (h.y >= 15).astype(float).to_numpy()
        lo, hi = ga.final_lo.to_numpy(), ga.final_hi.to_numpy()
        flagged = gp.stale.to_numpy()
        out[name] = {
            "pair_brier": brier(gp.p_final, p.miss),
            "pair_brier_flagged_rows": brier(gp.p_final[flagged], p.miss[flagged]) if flagged.any() else None,
            "pair_stale_share": float(flagged.mean()),
            "pair_mean_fallback_weight": float(gp.fallback_w.mean()),
            "held_out_coverage": float(((lo <= h.y) & (h.y <= hi)).mean()),
            "held_out_mean_width": float((hi - lo).mean()),
            "held_out_brier_p_late": brier(ga.p_late_final, o15),
            "arrival_stale_share": float(ga.stale.mean()),
        }  # fmt: skip
    nolab = p.act_arr_in.isna().to_numpy()
    flagged0 = Guard(design="binary", bin_min=12, bin_window=60).pairs(p).stale.to_numpy()
    out["label_check"] = {
        "share_of_rows_whose_inbound_arrival_is_not_in_the_archive": float(nolab.mean()),
        "share_of_flagged_rows_with_that": float(nolab[flagged0].mean())
        if flagged0.any()
        else None,
        "miss_rate_with_arrival_recorded": float(p.miss[~nolab].mean()),
        "miss_rate_without_arrival_recorded": float(p.miss[nolab].mean()) if nolab.any() else None,
    }
    out["pair_brier_rows_with_arrival_recorded"] = {
        n: brier(g.pairs(p).p_final[~nolab], p.miss[~nolab])
        for n, g in (("no_guard", none), ("stage6_first_guard", v1), ("frozen_guard", guard))
    }
    out["gap_bucket_alone"] = {"pair_brier": brier(p.p_bar, p.miss)}
    res["results"] = out
    res["oot_gap_second_look_done"] = True
    OUT.mkdir(exist_ok=True)
    GATE_SECOND.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
