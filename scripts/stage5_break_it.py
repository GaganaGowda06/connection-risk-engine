# ruff: noqa: E501
#!/usr/bin/env python3
"""Stage 5 failure tests, on VALIDATION pairs with the frozen final model (never the test block).

1 inbound cancelled mid-journey   2 late outbound that saves the connection
3 a hub with very few pairs       4 unknown MCT (the pair is dropped, never defaulted)
5 a missing hour of messages      (leave-hub-out on each hub is in stage5_fit and stage5_score)

    uv run python scripts/stage5_break_it.py --run-id RUN
"""

import argparse
import json
import shutil
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from rail_risk.baselines.split import network_split
from rail_risk.connections import baseline as BL
from rail_risk.connections import build as B
from rail_risk.connections import data as DD
from rail_risk.evaluation import _bin_stats

OUT = Path("reports/connections/stage5")
MODELS = Path("data/interim/stage5/models")
SCRATCH = Path("data/interim/stage5/_break_it")


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    cfg = B.load_config()
    fit = json.loads((OUT / "final_fit.json").read_text())
    pick = fit["option"]
    model = joblib.load(MODELS / f"{args.run_id}.joblib")
    tr = DD.labelled(DD.flag_groups(DD.load_block("train"), cfg))
    tr_ok = tr[~tr.held_group]
    bar = BL.fit(tr_ok)
    va_all = DD.flag_groups(DD.load_block("validation"), cfg)
    va = DD.labelled(va_all).reset_index(drop=True)
    va["p"] = model.predict(va)[f"{pick}_cal"].to_numpy()
    va["p_bar"] = BL.predict(bar, va)
    res: dict = {}
    # 1 inbound cancelled mid-journey: it ran, never reached the hub, so the label is a miss
    short = va[(va.in_ran) & va.act_arr_in.isna() & (va.horizon_min.isin([15, 30]))]
    res["1_inbound_cancelled_mid_journey"] = {
        "rows": int(len(short)),
        "pairs": int(short[["ssd", "in_rid", "out_rid"]].drop_duplicates().shape[0]),
        "mean_p_model_cancel_not_yet_known": float(short[~short.in_cancel_known].p.mean()),
        "n_cancel_not_yet_known": int((~short.in_cancel_known).sum()),
        "mean_p_model_cancel_known": float(short[short.in_cancel_known].p.mean()) if short.in_cancel_known.any() else None,
        "n_cancel_known": int(short.in_cancel_known.sum()),
        "mean_p_gap_bucket": float(short.p_bar.mean()), "actual_miss_rate": float(short.miss.mean()),
        "brier_model": brier(short.p, short.miss), "brier_gap_bucket": brier(short.p_bar, short.miss),
    }  # fmt: skip
    # 2 a late-running outbound saves a connection the schedule says is lost
    saved = va[
        (va.in_late_final_min > va.slack_min) & (va.miss == 0) & (va.out_late_final_min >= 3)
    ]
    res["2_late_outbound_saves_connection"] = {
        "rows": int(len(saved)),
        "mean_p_model": float(saved.p.mean()), "mean_p_gap_bucket": float(saved.p_bar.mean()),
        "brier_model": brier(saved.p, saved.miss), "brier_gap_bucket": brier(saved.p_bar, saved.miss),
        "note": "inbound later than the planned slack, yet the connection was made (label 0)",
    }  # fmt: skip
    # 3 a hub with very few pairs: 150 pairs of the smallest hub, scored cold with the global model
    hub = va.groupby("hub").size().idxmin()
    pairs = (
        va[va.hub == hub][["ssd", "in_rid", "out_rid"]]
        .drop_duplicates()
        .sample(150, random_state=0)
    )
    few = va.merge(pairs, on=["ssd", "in_rid", "out_rid"])
    b, e, _ = _bin_stats(few.p.to_numpy(), few.miss.to_numpy(float), 10)
    res["3_hub_with_150_pairs"] = {
        "hub": hub, "pairs": 150, "rows": int(len(few)), "brier_model": b, "ece": e,
        "brier_gap_bucket_pooled": brier(few.p_bar, few.miss),
        "note": "the pooled gap-bucket is used for a hub the bar never saw; ECE on 600 rows is noisy",
    }  # fmt: skip
    # 4 unknown MCT: a table without BHM drops every BHM pair, never defaults
    split, _ = network_split()
    day = split.validation[0]
    mct = B.load_mct_table()
    tc = pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
    full, st_full = B.build_day(Path("data/interim/pport"), day, cfg, mct, tc, 1)
    cut, st_cut = B.build_day(Path("data/interim/pport"), day, cfg, mct[mct.crs != "BHM"], tc, 1)
    res["4_unknown_mct"] = {
        "day": str(day), "bhm_rows_full": int((full.hub == "BHM").sum()),
        "bhm_rows_without_mct": int((cut.hub == "BHM").sum()),
        "other_hub_rows_unchanged": bool(len(full[full.hub != "BHM"]) == len(cut[cut.hub != "BHM"])),
        "dropped_unknown_mct": int(st_cut["dropped_unknown_mct"]),
    }  # fmt: skip
    # 5 a missing hour of messages: remove 09:00 to 10:00 UTC of stop_updates for one day
    root = SCRATCH / "pport"
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    for t in ("schedules", "schedule_points", "stop_actuals", "stop_updates"):
        src = Path("data/interim/pport") / t / f"service_date={day}"
        dst = root / t / f"service_date={day}"
        dst.mkdir(parents=True)
        for f in src.glob("*.parquet"):
            if t == "stop_updates":
                duckdb.execute(
                    f"COPY (SELECT * FROM read_parquet('{f}') WHERE NOT (message_time >= "  # noqa: S608
                    f"TIMESTAMPTZ '{day} 09:00:00+00' AND message_time < TIMESTAMPTZ '{day} 10:00:00+00')) "
                    f"TO '{dst / f.name}' (FORMAT PARQUET)"
                )
            else:
                shutil.copy(f, dst / f.name)
    gap, st_gap = B.build_day(root, day, cfg, mct, tc, 1)
    gap = DD.labelled(DD.flag_groups(gap, cfg)).reset_index(drop=True)
    full_l = DD.labelled(DD.flag_groups(full, cfg)).reset_index(drop=True)
    gap["p"] = model.predict(gap)[f"{pick}_cal"].to_numpy()
    full_l["p"] = model.predict(full_l)[f"{pick}_cal"].to_numpy()
    k = ["in_rid", "out_rid", "horizon_min"]
    m = full_l.merge(gap, on=k, suffixes=("_full", "_gap"))
    t = m.snap_time_full
    after = (t >= pd.Timestamp(f"{day} 10:00", tz="UTC")) & (
        t < pd.Timestamp(f"{day} 11:00", tz="UTC")
    )
    inside = (t >= pd.Timestamp(f"{day} 09:00", tz="UTC")) & (
        t < pd.Timestamp(f"{day} 10:00", tz="UTC")
    )
    res["5_missing_hour_0900_1000"] = {
        "day": str(day), "pairs_built_with_gap": int(len(gap)), "pairs_built_full": int(len(full_l)),
        "rows_scored_1000_1100": int(after.sum()),
        "brier_full": brier(m[after].p_full, m[after].miss_full),
        "brier_with_gap": brier(m[after].p_gap, m[after].miss_gap),
        "rows_1000_1100_state_changed": int((m[after].in_fc_late_s_full.fillna(-1) != m[after].in_fc_late_s_gap.fillna(-1)).sum()),
        "rows_inside_gap": int(inside.sum()),
        "brier_full_inside": brier(m[inside].p_full, m[inside].miss_full),
        "brier_with_gap_inside": brier(m[inside].p_gap, m[inside].miss_gap),
        "mean_abs_p_change_inside": float((m[inside].p_full - m[inside].p_gap).abs().mean()),
    }  # fmt: skip
    shutil.rmtree(SCRATCH)
    (OUT / "break_it.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
