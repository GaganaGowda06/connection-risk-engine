"""Frozen model artifacts for the service, built once from train and validation data only.

Arrival: the Stage 4 calibrated cold_noslack model (run 1e197857...) and its calibration run
c0a08ab4..., the Stage 2 baselines fitted on the train block (persistence, historical; the cold fit
is the base the model corrects), the global CQR scores for the adaptive ranges, the cold fastest
table. P(miss): the Stage 5 final run cd4ac6e9..., its gap-bucket bar. Nothing here reads the
test block, February or oot_week.
"""

from pathlib import Path

import joblib
import pandas as pd

ARRIVAL_RUN = "1e197857e96a41e8ad5d350d204307ed"
CALIBRATION_RUN = "c0a08ab4dfd8453380d3459a537ab27d"
PMISS_RUN = "cd4ac6e9d4fd4d999bb7216d450f600d"
ARTIFACTS = Path("data/interim/serve/artifacts.joblib")
CAL_DIR = Path("data/interim/stage4/calibrated")
MODELS5 = Path("data/interim/stage5/models")
ADAPTIVE_GAMMA = 0.05


def build() -> dict:
    from rail_risk.baselines import models
    from rail_risk.connections import baseline as BL
    from rail_risk.connections import build as CB
    from rail_risk.connections import data as DD
    from rail_risk.network_model import adaptive as A
    from rail_risk.network_model import calibrate as C
    from rail_risk.network_model import dataset as D
    from rail_risk.network_model import run as R
    from rail_risk.network_model import stage4 as S

    cal = C.Calibrated.load(CAL_DIR / ARRIVAL_RUN)
    rows, held, _split, _ = R.prepare_frames(include_test=False, cold=True)
    for k in ("train", "train_all_routes", "validation"):
        rows[k] = D.add_duration(rows[k])
    cold_fit = models.fit(rows["train"])
    hist_fit = models.fit(rows["train_all_routes"])
    pool = pd.concat(
        [
            pd.read_parquet(S.POOL_DIR / f"fold_pool_{ARRIVAL_RUN}.parquet"),
            S.final_pool_rows(
                cal.model,
                rows["train"],
                rows["validation"][~rows["validation"]["route"].isin(held)],
            ),
        ],
        ignore_index=True,
    )
    gs = A.GlobalScores.from_pool(pool)
    cfg = CB.load_config()
    tr = DD.labelled(DD.flag_groups(DD.load_block("train"), cfg))
    bar = BL.fit(tr[~tr.held_group])
    art = {
        "cold_fit": cold_fit, "hist_fit": hist_fit, "gs": gs, "bar": bar, "held": sorted(held),
        "arrival_run": ARRIVAL_RUN, "calibration_run": CALIBRATION_RUN, "pmiss_run": PMISS_RUN,
        "gamma": ADAPTIVE_GAMMA,
    }  # fmt: skip
    ARTIFACTS.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, ARTIFACTS)
    return art


def load() -> dict:
    if not ARTIFACTS.exists():
        return build()
    return joblib.load(ARTIFACTS)
