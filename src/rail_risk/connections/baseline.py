"""The gap-bucket bar for test 5, rescored on exactly the pairs the model is scored on.

P(miss) is the training miss rate inside buckets of the scheduled slack (gap minus MCT), per hub,
with the Stage 2 bucket edges and add-one smoothing. A hub the baseline was not fitted on (the
leave-hub-out folds) gets the pooled bucket rate. The baseline ignores the horizon, as in Stage 2.
"""

import numpy as np
import pandas as pd

from rail_risk.baselines.gap_bucket import BUCKET_EDGES


def bucket(slack: np.ndarray) -> np.ndarray:
    return np.digitize(slack, BUCKET_EDGES[1:-1])


def fit(train: pd.DataFrame) -> dict:
    b = bucket(train["slack_min"].to_numpy())
    y = train["miss"].to_numpy(float)
    out = {("*", -1): float((y.sum() + 1) / (len(y) + 2))}
    for k in np.unique(b):
        m = b == k
        out[("*", int(k))] = float((y[m].sum() + 1) / (m.sum() + 2))
    for hub, g in train.assign(_b=b, _y=y).groupby("hub"):
        for k, gg in g.groupby("_b"):
            out[(hub, int(k))] = float((gg["_y"].sum() + 1) / (len(gg) + 2))
    return out


def predict(model: dict, df: pd.DataFrame) -> np.ndarray:
    b = bucket(df["slack_min"].to_numpy())
    return np.array(
        [
            model.get((h, int(k)), model.get(("*", int(k)), model[("*", -1)]))
            for h, k in zip(df["hub"], b, strict=True)
        ]
    )
