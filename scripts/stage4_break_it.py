#!/usr/bin/env python3
"""Stage 4 break-it tests on the frozen calibrated model.
  1. coverage on the rows after a missing hour of messages (validation day)
  2. coverage on rows with nothing ahead (validation)
  3. coverage on the lst-cbg incident evening (2026-10-10, from the already scored test rows)
  4. a calibration group with very few rows (real Mondrian cells and a tiny pool)
    uv run python scripts/stage4_break_it.py --run-id RUN
Writes reports/network_model/stage4/break_it.json."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.baselines import models
from rail_risk.network_model import calibrate as C
from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F
from rail_risk.network_model import stage4 as S

OUT = Path("reports/network_model/stage4")
CAL_DIR = Path("data/interim/stage4/calibrated")


def cov(df: pd.DataFrame, lo, hi) -> float:
    y = df["target_min"].to_numpy()
    return round(float(((np.asarray(lo) <= y) & (y <= np.asarray(hi))).mean()), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    run_id = ap.parse_args().run_id
    cal = C.Calibrated.load(CAL_DIR / run_id)
    rows, held, split = S.prepare_calibration_frames(cold=True)
    train, val = rows["train"], rows["validation"]
    fit = models.fit(train)
    hv = val[val["route"].isin(held)].reset_index(drop=True)
    pred = cal.predict(hv, D.base_point(fit, hv))
    out: dict = {"run_id": run_id}

    # 1. a missing hour: rebuild one validation day without 10:00-11:00 UTC messages
    day = sorted(hv["ssd"].unique())[3]
    snaps = pd.read_parquet(D.SNAPSHOTS / f"service_date={day}" / "part.parquet")
    gap = (pd.Timestamp(f"{day} 10:00", tz="UTC"), pd.Timestamp(f"{day} 11:00", tz="UTC"))
    feats = F.build_day(D.PPORT, day, snaps, F.bank_holidays(), gap=gap)
    keys = snaps[[*F.KEY, "target_late_s", "has_target", "toc"]]
    feats = feats.assign(ssd=day)
    cut = D.prepare(feats.merge(keys, on=F.KEY, how="left"), pd.read_parquet(D.FASTEST_COLD))
    cut = D.add_duration(cut[cut["has_target"] & cut["route"].isin(held)].reset_index(drop=True))
    ref = hv[hv["ssd"] == day].merge(cut[F.KEY], on=F.KEY).reset_index(drop=True)
    cut = cut.merge(ref[F.KEY], on=F.KEY).reset_index(drop=True)
    ref_p = cal.predict(ref, D.base_point(fit, ref))
    cut_p = cal.predict(cut, D.base_point(fit, cut))
    sel = ref["snap_time"].between(gap[1], gap[1] + pd.Timedelta(hours=1)).to_numpy()
    out["missing_hour"] = {
        "day": str(day), "rows_after_gap": int(sel.sum()),
        "coverage_normal": cov(ref[sel], ref_p["lo"][sel], ref_p["hi"][sel]),
        "coverage_missing": cov(cut[sel], cut_p["lo"][sel], cut_p["hi"][sel]),
        "mean_width_normal": round(float((ref_p["hi"] - ref_p["lo"])[sel].mean()), 2),
        "mean_width_missing": round(float((cut_p["hi"] - cut_p["lo"])[sel].mean()), 2),
        "finite": bool(np.isfinite(cut_p[["point", "lo", "hi", "p_late"]]).all().all()),
    }  # fmt: skip

    # 2. nothing ahead
    none = ((hv["ahead_n"] == 0) & (hv["dest_n"] == 0) & (hv["next_n"] == 0)).to_numpy()
    out["nothing_ahead"] = {
        "rows": int(none.sum()),
        "coverage": cov(hv[none], pred["lo"][none], pred["hi"][none]),
        "coverage_other_rows": cov(hv[~none], pred["lo"][~none], pred["hi"][~none]),
    }

    # 3. the lst-cbg incident evening, from the scored test predictions
    tp = pd.read_parquet(CAL_DIR / run_id / "test_predictions.parquet")
    e = tp[(tp["route"] == "lst-cbg") & (tp["ssd"] == pd.Timestamp("2026-10-10").date())]
    incident = e["target_min"] >= 45
    out["lst_cbg_incident_evening"] = {
        "rows_that_day": int(len(e)), "incident_rows_late_45_plus": int(incident.sum()),
        "coverage_that_day": cov(e, e["lo"], e["hi"]),
        "coverage_incident_rows": cov(e[incident], e["lo"][incident], e["hi"][incident]),
        "coverage_other_rows": cov(e[~incident], e["lo"][~incident], e["hi"][~incident]),
        "coverage_lst_cbg_other_days": cov(
            tp[(tp["route"] == "lst-cbg") & ~tp["ssd"].eq(e["ssd"].iloc[0])],
            tp[(tp["route"] == "lst-cbg") & ~tp["ssd"].eq(e["ssd"].iloc[0])]["lo"],
            tp[(tp["route"] == "lst-cbg") & ~tp["ssd"].eq(e["ssd"].iloc[0])]["hi"],
        ),
    }  # fmt: skip

    # 4. calibration groups with very few rows (Mondrian cells) and a tiny pool
    pool = pd.read_parquet(S.POOL_DIR / f"fold_pool_{run_id}.parquet")
    g = C.group_frame(pool)
    sizes = g.groupby(list(g.columns)).size()
    mond = C.RangeCalibrator("mondrian").fit(pool)
    hg = C.group_frame(hv)
    fine = pd.MultiIndex.from_frame(hg[["h", "lb", "lc", "sb"]]).map(
        lambda k: sizes.get(k, 0) >= C.MIN_CELL
    )
    out["small_groups"] = {
        "finest_cells_in_pool": int(len(sizes)),
        "finest_cells_under_min_cell": int((sizes < C.MIN_CELL).sum()),
        "validation_held_out_rows_in_a_small_or_empty_finest_cell": int((~np.array(fine)).sum()),
        "share": round(float((~np.array(fine)).mean()), 4),
        "min_cell": C.MIN_CELL,
    }
    mp = D.base_point(fit, hv)
    raw = cal.model.predict(hv, mp)
    lo, hi = mond.apply(hv, raw["point"].to_numpy(), raw["lo"].to_numpy(), raw["hi"].to_numpy())
    out["small_groups"]["mondrian_coverage_all"] = cov(hv, lo, hi)
    out["small_groups"]["mondrian_coverage_rows_in_small_cells"] = (
        cov(hv[~np.array(fine)], lo[~np.array(fine)], hi[~np.array(fine)])
        if (~np.array(fine)).any()
        else None
    )
    try:
        C.RangeCalibrator("cqr").fit(pool.head(30))
        out["tiny_pool"] = "returned a range (should not happen)"
    except ValueError as exc:
        out["tiny_pool"] = f"raised: {exc}"
    (OUT / "break_it.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
