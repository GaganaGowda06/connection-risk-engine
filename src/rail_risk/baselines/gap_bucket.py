"""Gap-bucket P(miss) on interchange pairs: the fixed bar for test 5.

A pair is an inbound service reaching a hub and an outbound service leaving it. The planned gap
is the outbound public departure minus the inbound public arrival; a passenger plans a pair when
that gap is at least the minimum connection time (MCT) of the hub. A miss is the outbound train
leaving before the inbound arrival plus the MCT, or never running. The baseline is the training
miss rate inside buckets of (planned gap minus MCT), per hub. MCT is an input read from the RDG
MSN and TSI files (data/ref/mct.csv, ingest/mct.py); nothing is assumed here.
"""

import numpy as np
import pandas as pd

from rail_risk.ingest.mct import MctError, load_mct_table, mct_lookup

BUCKET_EDGES = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 60.0])
MctMissing = MctError

__all__ = ["BUCKET_EDGES", "MctMissing", "fit_buckets", "label_pairs", "load_mct_table"]


def pair_mct(pairs: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """MCT per pair from the real TOC codes of its two services (TSI over MSN); NaN when unknown."""
    cache: dict[tuple[str, str, str], float] = {}
    out = []
    for key in zip(pairs["hub"], pairs["in_toc"], pairs["out_toc"], strict=True):
        if key not in cache:
            v = mct_lookup(table, *key)
            cache[key] = float("nan") if v is None else v
        out.append(cache[key])
    return pd.Series(out, index=pairs.index, dtype="float64")


def label_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs: hub, mct_min, plan_arr_in, plan_dep_out, act_arr_in, act_dep_out (NaT = never left).

    Pairs with no MCT are dropped (fail closed), never given a default."""
    out = pairs[pairs["mct_min"].notna()].copy()
    m = out["mct_min"]
    out["slack_min"] = (out["plan_dep_out"] - out["plan_arr_in"]).dt.total_seconds() / 60.0 - m
    arrive = out["act_arr_in"] + pd.to_timedelta(m, unit="m")
    out["miss"] = (out["act_dep_out"].isna() | (out["act_dep_out"] < arrive)).astype(float)
    out = out[
        (out["slack_min"] >= 0) & (out["slack_min"] < BUCKET_EDGES[-1]) & out["act_arr_in"].notna()
    ]
    out["bucket"] = np.digitize(out["slack_min"], BUCKET_EDGES[1:-1])
    return out


def fit_buckets(train: pd.DataFrame) -> dict[tuple[str, int], float]:
    overall = (train["miss"].sum() + 1) / (len(train) + 2) if len(train) else 0.5
    out = {("*", -1): float(overall)}
    for (hub, b), g in train.groupby(["hub", "bucket"]):
        out[(hub, int(b))] = float((g["miss"].sum() + 1) / (len(g) + 2))
    return out


def predict_buckets(model: dict[tuple[str, int], float], pairs: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            model.get((h, int(b)), model[("*", -1)])
            for h, b in zip(pairs["hub"], pairs["bucket"], strict=True)
        ]
    )
