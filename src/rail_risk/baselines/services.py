"""Route services and their snapshot rows for one service date.

A route is data (RouteSpec: origin and destination TIPLOC sets), so nothing here names a route.
A service belongs to a route when its public calls reach the origin before the destination. The
timetable used is the first schedule message seen for the rid: it is the plan that existed
before the day, so it never depends on anything that happens after a snapshot time.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from rail_risk.clean import checks as ck
from rail_risk.ingest.pport_extract import train_pool_table

HORIZONS_MIN = (5, 15, 30, 60, 90, 120)
SCHEDULE_COLS = ["rid", "toc", "train_cat", "is_passenger", "message_time"]
POINT_COLS = ["rid", "seq", "tpl", "pt_arr", "pt_dep", "message_time"]
SERVICE_COLS = [
    "ssd", "rid", "route", "toc", "dest_tpl", "origin_tpl", "pt_arr", "pt_dep_origin",
    "schedule_seen", "later_versions_move_arrival",
]  # fmt: skip


def route_services(
    sched: pd.DataFrame, points: pd.DataFrame, specs: list[ck.RouteSpec], ssd: date
) -> pd.DataFrame:
    """One row per (rid, route) for passenger services, from the first schedule version."""
    if sched.empty or points.empty:
        return pd.DataFrame(columns=SERVICE_COLS)
    sched = sched[sched["is_passenger"].fillna(False) & ~sched["train_cat"].isin(ck.NON_TRAIN_CATS)]
    first = sched.sort_values("message_time").drop_duplicates("rid", keep="first")
    pts = points.merge(first[["rid", "message_time"]], on=["rid", "message_time"], how="inner")
    calls = pts[pts["pt_arr"].notna() | pts["pt_dep"].notna()]
    last_version = ck.latest_points(points, ck.latest_schedules(sched))
    last_arr = last_version.sort_values("seq").groupby(["rid", "tpl"])["pt_arr"].last()
    rows = []
    for spec in specs:
        o = calls[calls["tpl"].isin(spec.origin)].sort_values("seq").drop_duplicates("rid")
        d = calls[calls["tpl"].isin(spec.destination) & calls["pt_arr"].notna()]
        d = d.sort_values("seq").drop_duplicates("rid", keep="last")
        both = o[["rid", "seq", "tpl", "pt_dep"]].merge(
            d[["rid", "seq", "tpl", "pt_arr"]], on="rid", suffixes=("_o", "_d")
        )
        both = both[both["seq_o"] < both["seq_d"]]
        if both.empty:
            continue
        both["route"] = spec.id
        rows.append(both)
    if not rows:
        return pd.DataFrame(columns=SERVICE_COLS)
    out = pd.concat(rows, ignore_index=True).rename(
        columns={"tpl_o": "origin_tpl", "tpl_d": "dest_tpl", "pt_dep": "pt_dep_origin"}
    )
    out = out.merge(first[["rid", "toc", "message_time"]], on="rid").rename(
        columns={"message_time": "schedule_seen"}
    )
    later = last_arr.reindex(pd.MultiIndex.from_frame(out[["rid", "dest_tpl"]])).to_numpy()
    moved = (pd.Series(later, index=out.index) - out["pt_arr"]).abs() >= pd.Timedelta(minutes=1)
    out["later_versions_move_arrival"] = moved.fillna(False)
    out["ssd"] = ssd
    return out[SERVICE_COLS]


def snapshot_rows(services: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS_MIN) -> pd.DataFrame:
    """One row per service, route and horizon: T is the scheduled arrival minus the horizon."""
    h = pd.DataFrame({"horizon_min": list(horizons)})
    out = services.merge(h, how="cross")
    out["snap_time"] = out["pt_arr"] - pd.to_timedelta(out["horizon_min"], unit="m")
    # the schedule must have been known at T, or the service did not exist for the state yet
    out["schedule_known"] = out["schedule_seen"] <= out["snap_time"]
    return out


def day_services(
    root: Path, ssd: date, specs: list[ck.RouteSpec]
) -> pd.DataFrame:  # reads train_pool only
    sched = train_pool_table(root, "schedules", SCHEDULE_COLS, [ssd]).to_pandas()
    points = train_pool_table(root, "schedule_points", POINT_COLS, [ssd]).to_pandas()
    return route_services(sched, points, specs, ssd)
