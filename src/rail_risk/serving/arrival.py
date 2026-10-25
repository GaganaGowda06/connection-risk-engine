"""Arrival snapshots due in a time window, from whatever the store holds, with the frozen model.

The same code path as the batch pipeline (route services, snapshot state, features, base forecast,
calibrated model) but with no outcome tables: nothing here reads stop_actuals or any message that
is not in the store. A snapshot is due when its T = planned arrival minus the horizon falls in
[start, end).
"""

import gc
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from rail_risk.baselines.services import POINT_COLS, SCHEDULE_COLS, route_services, snapshot_rows
from rail_risk.baselines.state import UPDATE_COLS, state_duckdb
from rail_risk.ingest.pport_extract import train_pool_table
from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F

HORIZONS = (15, 60)


def route_services_filtered(root: Path, ssd: date, specs: list) -> pd.DataFrame:
    """Route services of one day, reading only the services that call at a route end (a superset
    of what route_services keeps), so the table in memory is a fraction of the day."""
    ends = sorted(set().union(*[s.origin | s.destination for s in specs]))
    files = sorted((root / "schedule_points").glob(f"service_date={ssd}/*.parquet"))
    if not files:
        return pd.DataFrame()
    con = duckdb.connect()
    lst = ", ".join(f"'{f}'" for f in files)
    tp = ", ".join(f"'{t}'" for t in ends)
    rids = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT rid FROM read_parquet([{lst}]) WHERE tpl IN ({tp})"  # noqa: S608
        ).fetchall()
    ]
    con.close()
    if not rids:
        return pd.DataFrame()
    where = ds.field("rid").isin(rids)
    sched = train_pool_table(root, "schedules", SCHEDULE_COLS, [ssd], where=where).to_pandas()
    points = train_pool_table(root, "schedule_points", POINT_COLS, [ssd], where=where).to_pandas()
    return route_services(sched, points, specs, ssd)


def due_arrival(
    root: Path,
    ssd: date,
    specs: list,
    start: pd.Timestamp,
    end: pd.Timestamp,
    art: dict,
    cal,
    fastest: pd.DataFrame,
    holidays: set,
    gap: tuple | None = None,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    services = route_services_filtered(root, ssd, specs)
    if services.empty:
        return pd.DataFrame()
    services_dur = services[["rid", "route", "pt_arr", "pt_dep_origin", "dest_tpl"]].copy()
    snaps = snapshot_rows(services, horizons)
    due = snaps[(snaps["snap_time"] >= start) & (snaps["snap_time"] < end)]
    known = due[due["schedule_known"]]
    if known.empty:
        return pd.DataFrame()
    updates = train_pool_table(
        root, "stop_updates", UPDATE_COLS, [ssd],
        where=ds.field("rid").isin(known["rid"].unique().tolist()),
    )  # fmt: skip
    if updates.num_rows == 0:
        return pd.DataFrame()
    if gap is not None:
        updates = F._without(updates, gap)
    state = state_duckdb(updates, known)
    del updates
    gc.collect()
    pa.default_memory_pool().release_unused()
    out = due.merge(state, on=["rid", "route", "horizon_min"], how="left")
    out["target_late_s"] = float("nan")
    out["has_target"] = False
    local = out["pt_arr"].dt.tz_convert("Europe/London")
    out["arr_hour_local"] = local.dt.hour
    out["weekday"] = local.dt.dayofweek
    del snaps, services
    gc.collect()
    feats = F.build_day(root, ssd, out, holidays, gap=gap)
    gc.collect()
    pa.default_memory_pool().release_unused()
    if feats.empty:
        return pd.DataFrame()
    keep = [*F.KEY, "ssd", "snap_time", "pt_arr", "toc", "target_late_s", "has_target"]
    extra = out[[c for c in keep if c in out and c not in feats]]
    keys = out[F.KEY + [c for c in extra if c not in F.KEY]]
    df = feats.merge(keys, on=F.KEY, how="left")
    df = D.prepare(df, fastest)
    dur = services_dur.drop_duplicates(["rid", "route"]).assign(
        sched_duration_min=lambda s: (s["pt_arr"] - s["pt_dep_origin"]).dt.total_seconds() / 60.0
    )[["rid", "route", "sched_duration_min"]]
    df = df.merge(dur, on=["rid", "route"], how="left")
    if "dest_tpl" not in df:
        df = df.merge(out[["rid", "route", "horizon_min", "dest_tpl"]], on=F.KEY, how="left")
    pred = cal.predict(df, D.base_point(art["cold_fit"], df))
    return df.assign(
        raw_lo=pred["raw_lo"].to_numpy(), raw_hi=pred["raw_hi"].to_numpy(),
        point=pred["point"].to_numpy(), static_lo=pred["lo"].to_numpy(),
        static_hi=pred["hi"].to_numpy(), p_late=pred["p_late"].to_numpy(),
    )  # fmt: skip
