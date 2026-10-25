#!/usr/bin/env python3
# ruff: noqa: E501  (analysis script with wide report dictionaries)
"""Stage 4 finish: break the adaptive ranges.
  a route with no history (first days of the block), the lst-cbg incident evening (how fast the
  range widens and recovers), a route silent for 3 days, and the update step too high or too low.
The step-size sweep uses validation only. The incident and silence tests reuse the already
second-scored test rows (diagnostics on those predictions, nothing is fitted).
    uv run python scripts/stage4b_break_it.py --run-id MODELRUN"""

import argparse
import json

import numpy as np
import pandas as pd

from rail_risk.network_model import adaptive as A
from rail_risk.network_model import stage4b as B


def cov(df, lo, hi) -> float:
    y = df["target_min"].to_numpy()
    return round(float(((np.asarray(lo) <= y) & (y <= np.asarray(hi))).mean()), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    run_id = ap.parse_args().run_id
    ch = B.load_choice()["choice"]
    method, gamma = ch["method"], ch["gamma"] or 0.0
    cal, rows, held, fit, gs = B.load_context(run_id)
    out: dict = {"method": method, "gamma": gamma}
    tp = pd.read_parquet(B.CAL_DIR / run_id / "test_adaptive_predictions.parquet")
    tp["ratio"] = (tp["adaptive_hi"] - tp["adaptive_lo"]) / (tp["static_hi"] - tp["static_lo"])
    tp["day"] = pd.to_datetime(tp["ssd"])

    # a: routes with no history: first day and first three days of the test block
    first = {}
    for route, g in tp.groupby("route"):
        d0 = g["day"].min()
        for label, k in (("day1", 1), ("days1_3", 3)):
            s = g[g["day"] < d0 + pd.Timedelta(days=k)]
            first.setdefault(route, {})[label] = {
                "n": len(s), "static": cov(s, s["static_lo"], s["static_hi"]),
                "adaptive": cov(s, s["adaptive_lo"], s["adaptive_hi"]),
                "width_ratio": round(float(s["ratio"].mean()), 3),
            }  # fmt: skip
    out["no_history_first_days"] = first

    # b: the lst-cbg incident evening of 2026-10-10 (rows at horizon 15 and 60, by arrival hour)
    e = tp[
        (tp["route"] == "lst-cbg")
        & (tp["ssd"] >= pd.Timestamp("2026-10-10").date())
        & (tp["ssd"] <= pd.Timestamp("2026-10-12").date())
    ].copy()
    e["hour"] = pd.to_datetime(e["pt_arr"]).dt.floor("2h")
    traj = e.groupby("hour").apply(
        lambda s: pd.Series({
            "n": len(s), "late45": int((s["target_min"] >= 45).sum()),
            "static_cov": cov(s, s["static_lo"], s["static_hi"]),
            "adaptive_cov": cov(s, s["adaptive_lo"], s["adaptive_hi"]),
            "width_ratio": round(float(s["ratio"].mean()), 3),
            "alpha": round(float(s["alpha"].mean()), 3),
        }),
        include_groups=False,
    )  # fmt: skip
    out["lst_cbg_incident_by_2h"] = {str(k): v for k, v in traj.round(3).to_dict("index").items()}
    daily = tp[tp["route"] == "lst-cbg"].groupby("ssd").apply(
        lambda s: pd.Series({"width_ratio": round(float(s["ratio"].mean()), 3),
                             "adaptive_cov": cov(s, s["adaptive_lo"], s["adaptive_hi"]),
                             "static_cov": cov(s, s["static_lo"], s["static_hi"])}),
        include_groups=False,
    )  # fmt: skip
    out["lst_cbg_daily"] = {str(k): v for k, v in daily.to_dict("index").items()}

    # c: a route silent for 3 days (all lst-cbg and bhm-liv rows removed 2026-10-12 to 10-14)
    test = rows["test"]
    hold = B.with_predictions(cal, fit, test[test["route"].isin(held)].reset_index(drop=True))
    full = A.run(hold, gs, method, gamma)
    drop = hold["route"].isin(["lst-cbg", "bhm-liv"]) & hold["ssd"].between(
        pd.Timestamp("2026-10-12").date(), pd.Timestamp("2026-10-14").date()
    )
    cut = A.run(hold[~drop], gs, method, gamma)
    after = (~drop) & (hold["ssd"] >= pd.Timestamp("2026-10-15").date())
    out["silent_3_days"] = {}
    for r in ("lst-cbg", "bhm-liv"):
        m = after & (hold["route"] == r)
        out["silent_3_days"][r] = {
            "rows_after": int(m.sum()),
            "coverage_full_stream": cov(hold[m], full.loc[m, "lo"], full.loc[m, "hi"]),
            "coverage_after_silence": cov(hold[m], cut.loc[m[m].index, "lo"], cut.loc[m[m].index, "hi"]),
            "static": cov(hold[m], hold.loc[m, "static_lo"], hold.loc[m, "static_hi"]),
        }  # fmt: skip

    # d: step size too low and too high, validation only
    val = rows["validation"]
    hv = B.with_predictions(cal, fit, val[val["route"].isin(held)].reset_index(drop=True))
    sweep = []
    for g in (0.0005, 0.002, 0.01, gamma, 0.2, 0.5):
        o = A.run(hv, gs, method, g)
        s = A.block_summary(hv, o["lo"], o["hi"])
        ratio = (o["hi"] - o["lo"]) / (hv["static_hi"] - hv["static_lo"])
        daily_ratio = ratio.groupby([hv["route"], hv["ssd"]]).mean()
        sweep.append({"gamma": g, "overall": round(s["overall"], 3), "worst15": s["worst15"],
                      "worst60": s["worst60"], "mean_width": round(s["mean_width"], 2),
                      "width_ratio_sd_within_route": round(float(daily_ratio.groupby(level=0).std().mean()), 3),
                      "max_width_ratio": round(float(ratio.max()), 2)})  # fmt: skip
    out["step_size_sweep_validation"] = sweep
    (B.OUT / "adaptive_break_it.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(
        json.dumps(
            {k: out[k] for k in ("silent_3_days", "step_size_sweep_validation")},
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
