"""Gap-bucket P(miss) on interchange pairs: the fixed bar for test 5.

A pair is an inbound service reaching a hub and an outbound service leaving it. The planned gap
is the outbound public departure minus the inbound public arrival; a passenger plans a pair when
that gap is at least the minimum connection time (MCT) of the hub. A miss is the outbound train
leaving before the inbound arrival plus the MCT, or never running. The baseline is the training
miss rate inside buckets of (planned gap minus MCT), per hub. MCT is an input read from a parsed
RDG file (data/ref/mct.csv: hub, minutes, source); nothing is assumed here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BUCKET_EDGES = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 60.0])
MCT_FILE = Path("data/ref/mct.csv")


class MctMissing(RuntimeError):
    pass


def load_mct(path: Path = MCT_FILE) -> dict[str, float]:
    if not path.exists():
        raise MctMissing(
            f"{path} not found: parse the RDG MSN and TSI files first (no MCT value is assumed)"
        )
    df = pd.read_csv(path)
    return dict(zip(df["hub"], df["minutes"].astype(float), strict=True))


def label_pairs(pairs: pd.DataFrame, mct: dict[str, float]) -> pd.DataFrame:
    """pairs: hub, plan_arr_in, plan_dep_out, act_arr_in, act_dep_out (NaT when it never ran)."""
    out = pairs.copy()
    m = out["hub"].map(mct)
    out["mct_min"] = m
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
