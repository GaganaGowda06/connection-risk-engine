"""Stage 4 finish helpers: score a block with the frozen model and the adaptive ranges."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.baselines import models
from rail_risk.network_model import adaptive as A
from rail_risk.network_model import calibrate as C
from rail_risk.network_model import dataset as D
from rail_risk.network_model import run as R
from rail_risk.network_model import stage4 as S

OUT = Path("reports/network_model/stage4b")
CAL_DIR = Path("data/interim/stage4/calibrated")
FEB = Path("data/interim/stage4/february")


def with_predictions(cal, fit, df: pd.DataFrame) -> pd.DataFrame:
    pred = cal.predict(df, D.base_point(fit, df))
    return df.assign(
        raw_lo=pred["raw_lo"].to_numpy(), raw_hi=pred["raw_hi"].to_numpy(),
        point=pred["point"].to_numpy(), static_lo=pred["lo"].to_numpy(),
        static_hi=pred["hi"].to_numpy(), p_late=pred["p_late"].to_numpy(),
    )  # fmt: skip


def load_choice() -> dict:
    return json.loads((OUT / "adaptive_choice.json").read_text())


def adaptive_block(
    df: pd.DataFrame,
    gs: A.GlobalScores,
    method: str,
    gamma: float,
    warm: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Adapted lo and hi for df. With `warm` (earlier rows with predictions and outcomes, e.g. the
    validation block) the stream starts with their outcomes already applied in time order."""
    if warm is None:
        return A.run(df, gs, method, gamma)
    both = pd.concat([warm, df], ignore_index=True)
    out = A.run(both, gs, method, gamma)
    return out.iloc[len(warm) :].set_axis(df.index)


def route_table(df: pd.DataFrame, lo: np.ndarray, hi: np.ndarray, static_lo, static_hi) -> list:
    y = df["target_min"].to_numpy()
    rows = []
    for route in sorted(df["route"].unique()):
        for h in (15, 60):
            m = ((df["route"] == route) & (df["horizon_min"] == h)).to_numpy()
            ok = (lo[m] <= y[m]) & (y[m] <= hi[m])
            ok0 = (static_lo[m] <= y[m]) & (y[m] <= static_hi[m])
            rows.append(
                {"route": route, "horizon_min": h, "n": int(m.sum()),
                 "static_coverage": float(ok0.mean()), "adaptive_coverage": float(ok.mean()),
                 "static_width": float((static_hi[m] - static_lo[m]).mean()),
                 "adaptive_width": float((hi[m] - lo[m]).mean())}
            )  # fmt: skip
    return rows


def load_context(run_id: str):
    cal = C.Calibrated.load(CAL_DIR / run_id)
    rows, held, split, _ = R.prepare_frames(include_test=True, cold=True)
    for k in ("train", "train_all_routes", "validation", "test"):
        rows[k] = D.add_duration(rows[k])
    fit = models.fit(rows["train"])
    gs = A.GlobalScores.from_pool(
        pd.concat(
            [
                pd.read_parquet(S.POOL_DIR / f"fold_pool_{run_id}.parquet"),
                S.final_pool_rows(
                    cal.model,
                    rows["train"],
                    rows["validation"][~rows["validation"]["route"].isin(held)],
                ),
            ],
            ignore_index=True,
        )
    )
    return cal, rows, held, fit, gs
