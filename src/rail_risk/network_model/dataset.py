"""Load the feature table, add the slack features and the Stage 2 base forecasts.

Rows come from data/interim/stage3/features (built from train_pool days only). The base forecasts
are the Stage 2 baselines, fitted on the train block exactly as `rail-risk evaluate-baselines`
does, so a model is compared with the same bar it is trained against.
"""

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.baselines import models
from rail_risk.baselines.split import Split, network_split
from rail_risk.config import load_routes
from rail_risk.network_model import features as F

FEAT_DIR = Path("data/interim/stage3/features")
FASTEST = Path("data/interim/stage3/fastest.parquet")
PPORT = Path("data/interim/pport")
LATE_MIN = models.LATE_MIN
CLIP = (-120.0, 240.0)  # minutes; Darwin forecasts occasionally carry a one-day error


def load(days: list[date], feat_dir: Path = FEAT_DIR) -> pd.DataFrame:
    parts = [feat_dir / f"service_date={d}" / "part.parquet" for d in days]
    frames = [pd.read_parquet(p) for p in parts if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ensure_fastest(
    all_days: list[date], train_days: list[date], rebuild: bool = False, path: Path = FASTEST
) -> pd.DataFrame:
    """Fastest observed run time per (from stop, event, destination), from train days only."""
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    pairs = pd.concat(
        [load([d])[F.INTERNAL] for d in all_days if (FEAT_DIR / f"service_date={d}").exists()]
    ).drop_duplicates()
    pairs = pairs.dropna()
    table = F.fastest_table(PPORT, train_days, pairs)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def prepare(df: pd.DataFrame, fastest: pd.DataFrame) -> pd.DataFrame:
    out = F.add_slack(df, fastest)
    out["target_min"] = out["target_late_s"].astype("float64") / 60.0
    # the column names the Stage 2 baselines read
    out["last_actual_late_s"] = out["late_now_min"] * 60.0
    out["fc_late_s"] = out["fc_late_min"] * 60.0
    out["schedule_known"] = True
    return out


def matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """The model input: FEATURES only, as floats, with lateness-like columns clipped."""
    names = cols or F.FEATURES
    F.assert_clean_feature_names(names)
    x = df[names].astype("float64").replace([np.inf, -np.inf], np.nan)
    for c in names:
        if c.endswith("_min") and c not in ("horizon_min",):
            x[c] = x[c].clip(*CLIP)
    return x


def split_rows(df: pd.DataFrame, split: Split, held_out: set[str]) -> dict[str, pd.DataFrame]:
    """train (train routes, train block), validation and test (all routes), each with targets."""
    ok = df[df["has_target"]]
    train_days, val_days, test_days = set(split.train), set(split.validation), set(split.test)
    return {
        "train": ok[ok["ssd"].isin(train_days) & ~ok["route"].isin(held_out)],
        "train_all_routes": ok[ok["ssd"].isin(train_days)],
        "validation": ok[ok["ssd"].isin(val_days)],
        "test": ok[ok["ssd"].isin(test_days)],
    }


def held_out_routes(path: Path = Path("config/routes.yaml")) -> set[str]:
    return {r.id for r in load_routes(path).routes if r.held_out and not r.legacy}


def base_forecasts(fit_rows: pd.DataFrame, rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stage 2 baselines fitted on fit_rows, predicted for rows (same columns as Stage 2)."""
    fitted = models.fit(fit_rows)
    return models.predict(fitted, rows)


def data_hash(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df[cols], index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def load_split() -> tuple[Split, set[str]]:
    split, _ = network_split()
    return split, held_out_routes()
