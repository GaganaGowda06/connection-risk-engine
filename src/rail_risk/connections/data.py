"""Load the Stage 5 pair tables, flag held-out service groups and assign tuning folds.

Reading rules: `load_block` reads only the block it is told to, from data/interim/stage5, which
the builder wrote from train_pool (February through the guarded reader). The model code asks for
train and validation until the stage is frozen.
"""

import hashlib

import numpy as np
import pandas as pd

from rail_risk.connections import build as B

DATA = B.PAIRS_DIR
N_FOLDS = 4


def load_block(block: str, days: list | None = None) -> pd.DataFrame:
    if block not in ("train", "validation", "test", "february"):
        raise ValueError(f"unknown block {block!r}")
    files = sorted((DATA / block).glob("service_date=*/part.parquet"))
    if days is not None:
        want = {str(d) for d in days}
        files = [f for f in files if f.parent.name.split("=")[1] in want]
    parts = [pd.read_parquet(f) for f in files]
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True)
    df["ssd"] = df["ssd"].astype(str)
    return df


def _fold(corr: str) -> int:
    return int(hashlib.md5(f"fold:{corr}".encode()).hexdigest(), 16) % N_FOLDS  # noqa: S324


def flag_groups(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    held = B.held_route_corridors()
    df = df.assign(in_corr=df["in_corr"].fillna("?|?"), out_corr=df["out_corr"].fillna("?|?"))
    mod, hold = cfg["group_mod"], cfg["group_hold"]
    uniq = pd.unique(pd.concat([df["in_corr"], df["out_corr"]]))
    is_held = {c: B.group_is_held(c, mod, hold, held) for c in uniq}
    fold = {c: _fold(c) for c in uniq}
    out = df.copy()
    out["in_held"] = out["in_corr"].map(is_held)
    out["out_held"] = out["out_corr"].map(is_held)
    out["held_group"] = out["in_held"] | out["out_held"]
    out["in_fold"] = out["in_corr"].map(fold)
    out["out_fold"] = out["out_corr"].map(fold)
    return out


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a P(miss) label: the outbound was not cancelled."""
    return df[(df["out_cancel"] == 0) & df["miss"].notna()]


def not_arrived(df: pd.DataFrame) -> pd.Series:
    return df["in_arr_late_s"].isna()


def fold_split(df: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Train mask (neither corridor in fold k) and test mask (inbound corridor in fold k)."""
    touch = (df["in_fold"] == k) | (df["out_fold"] == k)
    return (~touch).to_numpy(), (df["in_fold"] == k).to_numpy()
