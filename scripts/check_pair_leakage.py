#!/usr/bin/env python3
"""Leakage test 1 for Stage 5: pair state at T uses only messages at or before T.

For sampled real validation pairs the raw tables of the service date are cut to message_time <= T
first, and every state feature is recomputed in plain pandas as "the latest thing in what is left",
with no time condition of its own. The values must equal the DuckDB builder's, exactly.

    uv run python scripts/check_pair_leakage.py [--n 300]
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.connections import build as B
from rail_risk.connections import data as DD
from rail_risk.ingest.pport_extract import train_pool_table

ROOT = Path("data/interim/pport")
WIN = 12 * 3600
FEATS = ["in_arr_late_s", "in_fc_late_s", "in_last_late_s", "in_last_age_s", "in_cancel_known",
         "out_dep_late_s", "out_fc_late_s", "out_arr_late_s", "out_arr_actual",
         "out_last_late_s", "out_last_age_s", "out_cancel_known", "cong_recent_n",
         "cong_recent_late_min", "cong_cancel_n", "cong_sched_n"]  # fmt: skip


def secs(ts):
    return ts.timestamp()


def latest(su, rid, tpl, event, field, plan, T):
    d = su[(su.rid == rid) & (su.tpl == tpl) & (su.event == event) & su[field].notna()]
    d = d[(d[field] - plan).abs().dt.total_seconds() <= WIN]
    if d.empty:
        return np.nan
    d = d.sort_values(["message_time", field])
    return secs(d[field].iloc[-1]) - secs(plan)


def last_actual(su, rid, T):
    d = su[(su.rid == rid) & su.actual.notna()].copy()
    plan = np.where(
        d.event == "arr",
        d.pt_arr.fillna(d.wt_arr),
        np.where(d.event == "dep", d.pt_dep.fillna(d.wt_dep), d.wt_pass),
    )
    d["plan"] = pd.to_datetime(plan, utc=True)
    d = d[d.plan.notna()]
    d["late"] = (d.actual - d.plan).dt.total_seconds()
    d = d[d.late.abs() <= WIN]
    if d.empty:
        return np.nan, np.nan
    d = d.sort_values(["actual", "message_time", "tpl", "event"])
    r = d.iloc[-1]
    return float(r.late), secs(T) - secs(r.actual)


def cancel_known(sp, rid, tpl):
    d = sp[(sp.rid == rid) & (sp.tpl == tpl)]
    return False if d.empty else bool(d.sort_values("message_time").cancelled.iloc[-1])


def reference(tabs, r, tpl):
    T = r.snap_time
    su = tabs["su"][tabs["su"].message_time <= T]
    sp = tabs["sp"][tabs["sp"].message_time <= T]
    out = {}
    out["in_arr_late_s"] = latest(su, r.in_rid, tpl, "arr", "actual", r.plan_arr_in, T)
    out["in_fc_late_s"] = latest(su, r.in_rid, tpl, "arr", "est", r.plan_arr_in, T)
    out["in_last_late_s"], out["in_last_age_s"] = last_actual(su, r.in_rid, T)
    out["in_cancel_known"] = cancel_known(sp, r.in_rid, tpl)
    out["out_dep_late_s"] = latest(su, r.out_rid, tpl, "dep", "actual", r.plan_dep_out, T)
    out["out_fc_late_s"] = latest(su, r.out_rid, tpl, "dep", "est", r.plan_dep_out, T)
    first = tabs["sch"].sort_values("message_time").drop_duplicates("rid")
    fm = first[first.rid == r.out_rid].message_time.iloc[0]
    pts = tabs["sp_all"]
    pa = pts[
        (pts.rid == r.out_rid) & (pts.tpl == tpl) & (pts.message_time == fm) & pts.pt_arr.notna()
    ]
    if len(pa):
        plan_a = pa.pt_arr.iloc[0]
        a_act = latest(su, r.out_rid, tpl, "arr", "actual", plan_a, T)
        a_est = latest(su, r.out_rid, tpl, "arr", "est", plan_a, T)
        out["out_arr_actual"] = bool(not np.isnan(a_act))
        out["out_arr_late_s"] = a_act if not np.isnan(a_act) else a_est
    else:
        out["out_arr_actual"] = False
        out["out_arr_late_s"] = np.nan
    out["out_last_late_s"], out["out_last_age_s"] = last_actual(su, r.out_rid, T)
    out["out_cancel_known"] = cancel_known(sp, r.out_rid, tpl)
    # hub context
    arr = su[(su.tpl == tpl) & (su.event == "arr") & su.actual.notna()].copy()
    arr["plan"] = arr.pt_arr.fillna(arr.wt_arr)
    arr = arr[arr.plan.notna()]
    arr["late"] = (arr.actual - arr.plan).dt.total_seconds()
    arr = arr[arr.late.abs() <= WIN].sort_values("message_time")
    f = arr.drop_duplicates("rid", keep="first")
    f = f[f.actual > T - pd.Timedelta(minutes=30)]
    out["cong_recent_n"] = len(f)
    out["cong_recent_late_min"] = f.late.mean() / 60.0 if len(f) else np.nan
    c = sp[(sp.tpl == tpl) & sp.cancelled].groupby("rid").message_time.min()
    out["cong_cancel_n"] = int(((c <= T) & (c > T - pd.Timedelta(minutes=60))).sum())
    seen = first.set_index("rid").message_time
    cand = tabs["sp_all"][
        (tabs["sp_all"].tpl == tpl) & tabs["sp_all"].message_time.eq(tabs["sp_all"].rid.map(seen))
    ]
    cand = cand[cand.pt_arr.notna() | cand.pt_dep.notna()]
    cand = cand[cand.rid.isin(first.rid)]
    t = cand.pt_dep.fillna(cand.pt_arr)
    ok = (t >= r.plan_dep_out - pd.Timedelta(minutes=30)) & (
        t <= r.plan_dep_out + pd.Timedelta(minutes=30)
    )
    ok &= cand.rid.map(seen) <= T
    out["cong_sched_n"] = int(ok.sum())
    return out


def same(a, b):
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--days", type=int, default=6)
    args = ap.parse_args()
    cfg = B.load_config()
    val = DD.load_block("validation")
    rng = np.random.default_rng(20261015)
    days = sorted(rng.choice(sorted(val.ssd.unique()), args.days, replace=False))
    checked, bad, per = 0, [], {f: 0 for f in FEATS}
    nonnull = {f: 0 for f in FEATS}
    for d in days:
        dd = date.fromisoformat(d)
        tabs = {
            "su": train_pool_table(
                ROOT,
                "stop_updates",
                [
                    "rid",
                    "tpl",
                    "event",
                    "pt_arr",
                    "pt_dep",
                    "wt_arr",
                    "wt_dep",
                    "wt_pass",
                    "est",
                    "actual",
                    "message_time",
                ],
                [dd],
            ).to_pandas(),
            "sp": train_pool_table(
                ROOT, "schedule_points", ["rid", "tpl", "cancelled", "message_time"], [dd]
            ).to_pandas(),
            "sp_all": train_pool_table(
                ROOT, "schedule_points", ["rid", "tpl", "pt_arr", "pt_dep", "message_time"], [dd]
            ).to_pandas(),
            "sch": train_pool_table(
                ROOT, "schedules", ["rid", "message_time", "is_passenger", "train_cat"], [dd]
            ).to_pandas(),
        }
        tabs["sch"] = tabs["sch"][
            tabs["sch"].is_passenger.fillna(False)
            & ~tabs["sch"].train_cat.fillna("").isin(["BR", "BS", "SS"])
        ]
        tpls = (
            B.hub_tiplocs(cfg["hubs"], pd.read_parquet("data/interim/ref/tiploc_crs.parquet"))
            .set_index("hub")
            .tpl
        )
        rows = val[val.ssd == d]
        pick = rows.sample(min(len(rows), args.n // args.days), random_state=1)
        for r in pick.itertuples():
            ref = reference(tabs, r, tpls[r.hub])
            checked += 1
            for f in FEATS:
                got = getattr(r, f)
                nonnull[f] += int(not pd.isna(got))
                if not same(got, ref[f]):
                    per[f] += 1
                    if len(bad) < 10:
                        bad.append(
                            {
                                "day": d,
                                "in": r.in_rid,
                                "out": r.out_rid,
                                "h": r.horizon_min,
                                "feat": f,
                                "built": None if pd.isna(got) else float(got),
                                "ref": None if pd.isna(ref[f]) else float(ref[f]),
                            }
                        )
    res = {
        "checked": checked,
        "days": days,
        "mismatches": per,
        "total_mismatches": sum(per.values()),
        "nonnull": nonnull,
        "examples": bad,
    }
    Path("reports/connections/stage5").mkdir(parents=True, exist_ok=True)
    Path("reports/connections/stage5/pair_leakage.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    raise SystemExit(1 if res["total_mismatches"] else 0)


if __name__ == "__main__":
    main()
