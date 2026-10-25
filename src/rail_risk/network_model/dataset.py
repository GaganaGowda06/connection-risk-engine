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
FASTEST_COLD = Path("data/interim/stage3/fastest_cold.parquet")
PPORT = Path("data/interim/pport")
SNAPSHOTS = Path("data/interim/stage2/snapshots")
LATE_MIN = models.LATE_MIN
CLIP = (-120.0, 240.0)  # minutes; Darwin forecasts occasionally carry a one-day error


def load(days: list[date], feat_dir: Path = FEAT_DIR) -> pd.DataFrame:
    parts = [feat_dir / f"service_date={d}" / "part.parquet" for d in days]
    frames = [pd.read_parquet(p) for p in parts if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ensure_fastest(
    all_days: list[date],
    train_days: list[date],
    rebuild: bool = False,
    cold: bool = False,
) -> pd.DataFrame:
    """Fastest observed run time per (from stop, event, destination), from train days only. In cold
    start the held-out routes' own services are left out, so they get no route-specific slack."""
    path = FASTEST_COLD if cold else FASTEST
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    frames = [load([d]) for d in all_days if (FEAT_DIR / f"service_date={d}").exists()]
    allrows = pd.concat(frames)
    pairs = allrows[F.INTERNAL].drop_duplicates().dropna()
    exclude = None
    if cold:
        held = held_out_routes()
        in_train = allrows["ssd"].isin(set(train_days))
        exclude = set(allrows.loc[in_train & allrows["route"].isin(held), "rid"])
    table = F.fastest_table(PPORT, train_days, pairs, exclude_rids=exclude)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def add_slot_history(df: pd.DataFrame) -> pd.DataFrame:
    """slot_* features: lateness quantities of the same route and arrival hour over the previous
    SLOT_DAYS service dates present in df, strictly before the row's own date. Needs targets of
    earlier days only, so a row never sees its own day or later."""
    svc = df[df["has_target"]].drop_duplicates(["rid", "route"])
    keys = df[["route", "arr_hour_local", "ssd"]].drop_duplicates()
    stats = []
    for (route, hour), g in svc.groupby(["route", "arr_hour_local"]):
        by_day = {d: v["target_min"].to_numpy() for d, v in g.groupby("ssd")}
        days = sorted(by_day)
        want = keys[(keys["route"] == route) & (keys["arr_hour_local"] == hour)]["ssd"]
        for d in want:
            lo = d - pd.Timedelta(days=F.SLOT_DAYS).to_pytimedelta()
            window = [by_day[x] for x in days if lo <= x < d]
            y = np.concatenate(window) if window else np.array([])
            ok = len(y) >= F.SLOT_MIN_N
            stats.append(
                {
                    "route": route, "arr_hour_local": hour, "ssd": d,
                    "slot_median_min": float(np.median(y)) if ok else np.nan,
                    "slot_p90_min": float(np.quantile(y, 0.9)) if ok else np.nan,
                    "slot_late15_share": float((y >= LATE_MIN).mean()) if ok else np.nan,
                    "slot_n": float(len(y)),
                }
            )  # fmt: skip
    slot_cols = ["slot_median_min", "slot_p90_min", "slot_late15_share", "slot_n"]
    st = pd.DataFrame(stats, columns=["route", "arr_hour_local", "ssd", *slot_cols])
    base = df.drop(columns=[c for c in F.HISTORY if c in df], errors="ignore")
    out = base.merge(st, on=["route", "arr_hour_local", "ssd"], how="left")
    return add_recent(out)


def add_recent(df: pd.DataFrame) -> pd.DataFrame:
    """recent_*: over the previous RECENT_DAYS service dates of the same route (and horizon for the
    recovery), the median minutes gained or lost between T and arrival and the median final
    lateness. Strictly earlier dates only."""
    ok = df[df["has_target"]].copy()
    ok["rec"] = ok["target_min"] - ok["late_now_min"].fillna(0.0)
    rows = []
    for (route, horizon), g in ok.groupby(["route", "horizon_min"]):
        by_day = {d: v for d, v in g.groupby("ssd")}
        days = sorted(by_day)
        want = df[(df["route"] == route) & (df["horizon_min"] == horizon)]["ssd"].unique()
        for d in want:
            lo = d - pd.Timedelta(days=F.RECENT_DAYS).to_pytimedelta()
            parts = [by_day[x] for x in days if lo <= x < d]
            n = sum(len(p) for p in parts)
            good = n >= F.RECENT_MIN_N
            cat = pd.concat(parts) if parts else None
            rows.append(
                {
                    "route": route, "horizon_min": horizon, "ssd": d,
                    "recent_recovery_min": float(cat["rec"].median()) if good else np.nan,
                    "recent_final_min": float(cat["target_min"].median()) if good else np.nan,
                    "recent_n": float(n),
                }
            )  # fmt: skip
    cols = ["route", "horizon_min", "ssd", "recent_recovery_min", "recent_final_min", "recent_n"]
    return df.merge(
        pd.DataFrame(rows, columns=cols), on=["route", "horizon_min", "ssd"], how="left"
    )


def prepare(df: pd.DataFrame, fastest: pd.DataFrame) -> pd.DataFrame:
    out = F.add_slack(df, fastest)
    out["target_min"] = out["target_late_s"].astype("float64") / 60.0
    out = add_slot_history(out)
    # the column names the Stage 2 baselines read
    out["last_actual_late_s"] = out["late_now_min"] * 60.0
    out["fc_late_s"] = out["fc_late_min"] * 60.0
    out["schedule_known"] = True
    return out


def add_duration(df: pd.DataFrame, snap_dir: Path = SNAPSHOTS) -> pd.DataFrame:
    """sched_duration_min: scheduled minutes from the route origin to the destination (a
    transferable length measure from the timetable alone, used for calibration groups)."""
    parts = []
    for d in sorted(df["ssd"].unique()):
        f = snap_dir / f"service_date={d}" / "part.parquet"
        s = pd.read_parquet(f, columns=["rid", "route", "pt_dep_origin", "pt_arr"])
        s = s.drop_duplicates(["rid", "route"])
        s["sched_duration_min"] = (s["pt_arr"] - s["pt_dep_origin"]).dt.total_seconds() / 60.0
        parts.append(s[["rid", "route", "sched_duration_min"]])
    dur = pd.concat(parts, ignore_index=True)
    return df.merge(dur, on=["rid", "route"], how="left")


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


def base_point(fitted: "models.Fitted", df: pd.DataFrame, warm: bool = False) -> np.ndarray:
    """The point the residual model corrects. Default: the Stage 2 shrunk persistence. Warm: the
    lateness now plus the route's median recovery over its previous RECENT_DAYS days (zero when
    there is not enough history), which follows a route whose running drifts."""
    if warm:
        return (df["late_now_min"].fillna(0.0) + df["recent_recovery_min"].fillna(0.0)).to_numpy()
    return models.predict(fitted, df)["shrunk_persistence"]["point"].to_numpy()


def data_hash(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df[cols], index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def load_split() -> tuple[Split, set[str]]:
    split, _ = network_split()
    return split, held_out_routes()
