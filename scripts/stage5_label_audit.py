#!/usr/bin/env python3
# ruff: noqa: E501
"""Pair label audit (my decision, 2026-10-26): how many pair rows now labelled miss would become
UNKNOWN under the rule written in DECISIONS.md first: an inbound whose arrival at the hub is missing
from the archive, with no cancellation and no deactivation message for that service, is unknown.

Counts only, nothing is scored or fitted. train, validation and test come from the stored stage 5
pair files (train is the 1 in 8 sample the models used); February from its stored files; oot_week is
rebuilt day by day, because no stored pair file exists. February and oot_week are read only through
the guarded reader with the purpose pair_label_audit, once (the result file is the marker).

    uv run python scripts/stage5_label_audit.py
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from rail_risk.connections import build as CB
from rail_risk.ingest.pport_extract import GATE_AUDIT, reading_split, train_pool_table

ROOT = Path("data/interim/pport")
OUT = GATE_AUDIT
COLS = ["hub", "in_rid", "miss", "out_cancel", "act_arr_in", "in_cancel_final"]


def deact_rids(ssd: date) -> set:
    t = train_pool_table(ROOT, "deactivations", columns=["rid"], service_dates=[ssd])
    return set(t["rid"].to_pylist()) if t.num_rows else set()


def classify(df: pd.DataFrame, deact: set) -> pd.DataFrame:
    """Per hub counts for one day of labelled pair rows."""
    d = df[df["miss"].notna()].copy()
    no_arr = d["act_arr_in"].isna()
    cancel = d["in_cancel_final"].fillna(False).astype(bool)
    dead = d["in_rid"].isin(deact)
    d["labelled"] = 1
    d["miss_now"] = (d["miss"] == 1).astype(int)
    d["missing_arrival"] = (d["miss"] == 1) & no_arr
    d["missing_and_cancelled"] = d["missing_arrival"] & cancel
    d["missing_and_deactivated_not_cancelled"] = d["missing_arrival"] & ~cancel & dead
    d["to_unknown"] = d["missing_arrival"] & ~cancel & ~dead
    d["variant_ignoring_deactivation"] = d["missing_arrival"] & ~cancel  # NOT my rule
    d["inbound_has_deactivation"] = dead
    g = d.groupby("hub")[
        [
            "labelled",
            "miss_now",
            "missing_arrival",
            "missing_and_cancelled",
            "missing_and_deactivated_not_cancelled",
            "to_unknown",
        ]
    ].sum()
    return g.astype(int)


def stored_block(block: str, deact_fn) -> pd.DataFrame:
    parts = []
    for d in sorted(Path(f"data/interim/stage5/{block}").glob("service_date=*/part.parquet")):
        ssd = date.fromisoformat(d.parent.name.split("=")[1])
        df = pd.read_parquet(d, columns=COLS)
        parts.append(classify(df, deact_fn(ssd)))
    return pd.concat(parts).groupby(level=0).sum()


def finish(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["miss_remaining"] = g["miss_now"] - g["to_unknown"]
    g["unknown_share_of_labelled_pct"] = (100 * g["to_unknown"] / g["labelled"]).round(3)
    g["unknown_share_of_misses_pct"] = (100 * g["to_unknown"] / g["miss_now"]).round(2)
    g["variant_share_of_labelled_pct"] = (
        100 * g["variant_ignoring_deactivation"] / g["labelled"]
    ).round(3)
    g["deactivation_share_of_labelled_pct"] = (
        100 * g["inbound_has_deactivation"] / g["labelled"]
    ).round(2)
    return g


def oot_block() -> pd.DataFrame:
    cfg = CB.load_config()
    mct, tc = CB.load_mct_table(), pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
    parts = []
    for i in range(7):
        d = date(2026, 10, 17 + i)
        df, _ = CB.build_day(ROOT, d, cfg, mct, tc, 1)
        parts.append(classify(df, deact_rids(d)))
    return pd.concat(parts).groupby(level=0).sum()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"already done: {OUT}")
    res: dict = {}
    for block in ("train", "validation", "test"):
        res[block] = finish(stored_block(block, deact_rids))
        print(block, flush=True)
        print(res[block].to_string(), flush=True)
    with reading_split("winter_test", "pair_label_audit"):
        res["february"] = finish(stored_block("february", deact_rids))
    print("february")
    print(res["february"].to_string(), flush=True)
    with reading_split("oot_week", "pair_label_audit"):
        res["oot_week"] = finish(oot_block())
    print("oot_week")
    print(res["oot_week"].to_string(), flush=True)
    out = {
        "rule": "inbound arrival missing from the archive, no cancellation and no deactivation message => unknown (excluded)",
        "note": "train is the 1 in 8 sample used by stage 5; nothing was scored or fitted",
        "counts": {k: v.reset_index().to_dict("records") for k, v in res.items()},
        "label_audit_done": True,
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
