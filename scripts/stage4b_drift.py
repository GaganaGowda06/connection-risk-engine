#!/usr/bin/env python3
"""Stage 4 finish, step 2: is the bhm-liv failure seasonal drift outside the training dates?

Weekly median residual and coverage of the frozen static model for each held-out route across the
train, validation, test and February blocks (held-out routes were never trained on, so their
train-block rows are out-of-route too); weekly running pattern and scheduled duration of every
route; and a drift comparison across the routes of the Birmingham hub.
    uv run python scripts/stage4b_drift.py --run-id MODELRUN
Diagnostic only: it fits and calibrates nothing. Writes reports/network_model/stage4b/drift_*.
"""

import argparse
from pathlib import Path

import pandas as pd

from rail_risk.baselines import models
from rail_risk.config import load_routes
from rail_risk.network_model import calibrate as C
from rail_risk.network_model import dataset as D
from rail_risk.network_model import run as R

OUT = Path("reports/network_model/stage4b")
CAL_DIR = Path("data/interim/stage4/calibrated")
FEB = Path("data/interim/stage4/february")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    run_id = ap.parse_args().run_id
    OUT.mkdir(parents=True, exist_ok=True)
    cal = C.Calibrated.load(CAL_DIR / run_id)
    rows, held, split, _ = R.prepare_frames(include_test=True, cold=True)
    fit = models.fit(rows["train"])
    frames = {}
    for name in ("train_all_routes", "validation", "test"):
        frames[name] = D.add_duration(rows[name])
    frames["train"] = frames.pop("train_all_routes")
    fastest = pd.read_parquet(D.FASTEST_COLD)
    feb = D.prepare(
        D.load(sorted({p for p in range(1)} and []) or _feb_days(), FEB / "features"), fastest
    )
    feb = D.add_duration(feb, FEB / "snapshots")
    frames["february"] = feb
    parts = []
    for block, df in frames.items():
        df = df[df["has_target"] & (df["horizon_min"] == 15)].reset_index(drop=True)
        pred = cal.predict(df, D.base_point(fit, df))
        df = df.assign(
            point=pred["point"], lo=pred["lo"], hi=pred["hi"], block=block,
            resid=df["target_min"] - pred["point"],
            covered=(
                (pred["lo"] <= df["target_min"]) & (df["target_min"] <= pred["hi"])
            ).astype(float),
            recovery=df["target_min"] - df["late_now_min"].fillna(0.0),
        )  # fmt: skip
        parts.append(df)
    allr = pd.concat(parts, ignore_index=True)
    allr["week"] = pd.to_datetime(allr["ssd"]).dt.to_period("W").dt.start_time
    h = allr[allr["route"].isin(held)]
    weekly = (
        h.groupby(["route", "block", "week"])
        .agg(n=("resid", "size"), median_resid=("resid", "median"), coverage=("covered", "mean"),
             median_final_late=("target_min", "median"), median_recovery=("recovery", "median"))
        .reset_index()
    )  # fmt: skip
    weekly.to_csv(OUT / "drift_weekly_held_out.csv", index=False)
    # drift by route: change in median recovery between a reference span and the last weeks
    ref = allr[
        (allr["ssd"] >= pd.Timestamp("2026-08-03").date())
        & (allr["ssd"] <= pd.Timestamp("2026-09-06").date())
    ]
    late = allr[
        (allr["ssd"] >= pd.Timestamp("2026-09-21").date())
        & (allr["ssd"] <= pd.Timestamp("2026-10-16").date())
    ]
    summ = pd.DataFrame(
        {
            "recovery_aug_to_sep6": ref.groupby("route")["recovery"].median(),
            "recovery_sep21_to_oct16": late.groupby("route")["recovery"].median(),
            "final_late_aug_to_sep6": ref.groupby("route")["target_min"].median(),
            "final_late_sep21_to_oct16": late.groupby("route")["target_min"].median(),
        }
    )
    summ["recovery_change"] = summ["recovery_sep21_to_oct16"] - summ["recovery_aug_to_sep6"]
    summ["final_late_change"] = summ["final_late_sep21_to_oct16"] - summ["final_late_aug_to_sep6"]
    routes = {r.id: r.hub for r in load_routes("config/routes.yaml").routes if not r.legacy}
    summ["hub"] = summ.index.map(routes)
    summ["held_out"] = summ.index.isin(held)
    summ = summ.sort_values("recovery_change")
    summ.to_csv(OUT / "drift_by_route.csv")
    # scheduled duration per week and the day of any step (timetable change)
    sd = allr.drop_duplicates(["rid", "route"])
    dur = sd.groupby(["route", "week"])["sched_duration_min"].median().unstack("week")
    dur.to_csv(OUT / "drift_sched_duration_weekly.csv")
    daily = sd[sd["route"] == "bhm-liv"].groupby("ssd")["sched_duration_min"].median().sort_index()
    pd.set_option("display.width", 250)
    print(summ.round(2).to_string())
    print("bhm-liv scheduled duration by week (minutes):")
    print(dur.loc["bhm-liv"].dropna().tail(14).round(1).to_string())
    print("bhm-liv daily scheduled duration range over last 60 days:",
          daily.tail(60).min(), daily.tail(60).max())  # fmt: skip
    for r in ("bhm-liv", "lst-cbg"):
        w = weekly[weekly["route"] == r].sort_values("week")
        print(r)
        print(
            w[["block", "week", "n", "median_resid", "coverage", "median_recovery"]]
            .round(2)
            .to_string(index=False)
        )


def _feb_days():
    from datetime import date

    return [date(2026, 2, d) for d in range(1, 29)]


if __name__ == "__main__":
    main()
