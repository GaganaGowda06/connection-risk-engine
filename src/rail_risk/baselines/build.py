"""Build the Stage 2 snapshot table, one service date at a time, from train_pool only.

Each row is one service, route and horizon. State comes from messages at or before the snapshot
time (state.py); the target is the actual arrival lateness at the destination. Days with no
extracted data are logged and skipped, so a missing day shows up as a missing day, not an error.
"""

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

from rail_risk.baselines.services import day_services, snapshot_rows
from rail_risk.baselines.state import ENGINES, UPDATE_COLS
from rail_risk.clean import checks as ck
from rail_risk.ingest.pport_extract import train_pool_table
from rail_risk.logging import get_logger

SNAP_DIR = "snapshots"


MAX_TARGET_GAP = pd.Timedelta(hours=12)


def day_targets(root: Path, ssd: date) -> pd.DataFrame:
    """Every distinct actual arrival per (rid, stop). The extract can hold a second row for the same
    rid and stop dated one day later (a day-roll error in the feed), so the choice between them is
    made against the planned time in pick_targets, not here."""
    a = train_pool_table(root, "stop_actuals", ["rid", "tpl", "event", "actual"], [ssd])
    if a.num_rows == 0:
        return pd.DataFrame(columns=["rid", "tpl", "actual"])
    df = a.to_pandas()
    df = df[(df["event"] == "arr") & df["actual"].notna()]
    return df[["rid", "tpl", "actual"]].drop_duplicates()


def pick_targets(out: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Attach the actual arrival nearest the planned arrival; none when the nearest is over 12
    hours away. out has rid, route, horizon_min, dest_tpl, pt_arr; targets has rid, dest_tpl,
    actual_arr."""
    keys = ["rid", "route", "horizon_min"]
    both = out.merge(targets, on=["rid", "dest_tpl"], how="left")
    gap = (both["actual_arr"] - both["pt_arr"]).abs()
    both = both.assign(_gap=gap).sort_values([*keys, "_gap"], na_position="last")
    both = both.drop_duplicates(keys, keep="first")
    far = both["_gap"] > MAX_TARGET_GAP
    both.loc[far, "actual_arr"] = pd.NaT
    return both.drop(columns="_gap").sort_index()


def build_day(
    root: Path, ssd: date, specs: list[ck.RouteSpec], engine: str = "duckdb"
) -> pd.DataFrame | None:
    services = day_services(root, ssd, specs)
    if services.empty:
        return None
    snaps = snapshot_rows(services)
    updates = train_pool_table(root, "stop_updates", UPDATE_COLS, [ssd])
    if updates.num_rows == 0:
        return None
    known = snaps[snaps["schedule_known"]]
    state = ENGINES[engine](updates, known)
    out = snaps.merge(state, on=["rid", "route", "horizon_min"], how="left")
    targets = day_targets(root, ssd).rename(columns={"tpl": "dest_tpl", "actual": "actual_arr"})
    out = pick_targets(out, targets)
    out["target_late_s"] = (out["actual_arr"] - out["pt_arr"]).dt.total_seconds()
    out["has_target"] = out["target_late_s"].notna()
    local = out["pt_arr"].dt.tz_convert("Europe/London")
    out["arr_hour_local"] = local.dt.hour
    out["weekday"] = local.dt.dayofweek
    return out


def build_range(
    root: Path,
    days: list[date],
    specs: list[ck.RouteSpec],
    out_dir: Path,
    engine: str = "duckdb",
    rebuild: bool = False,
) -> dict:
    """Write one Parquet file per service date under out_dir; return a log of what happened."""
    log = get_logger("baselines_build")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"built": [], "skipped_existing": [], "missing_days": [], "rows": 0}
    for d in days:
        dest = out_dir / f"service_date={d}" / "part.parquet"
        if dest.exists() and not rebuild:
            report["skipped_existing"].append(str(d))
            continue
        t0 = time.monotonic()
        df = build_day(root, d, specs, engine)
        if df is None:
            log.warning("day_missing", day=str(d))
            report["missing_days"].append(str(d))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        report["built"].append(str(d))
        report["rows"] += len(df)
        log.info("day_built", day=str(d), rows=len(df), seconds=round(time.monotonic() - t0, 1))
    return report


def load_snapshots(out_dir: Path, days: list[date]) -> pd.DataFrame:
    parts = [out_dir / f"service_date={d}" / "part.parquet" for d in days]
    frames = [pd.read_parquet(p) for p in parts if p.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
