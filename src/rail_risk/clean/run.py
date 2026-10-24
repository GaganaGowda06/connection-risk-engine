"""Run the Stage 1 checks over the extracted tables, one service date at a time.

The route list, the station map and the data root are all arguments; nothing is named here.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from rail_risk.clean import checks as ck
from rail_risk.config import RouteSet

DUP_KEYS = [
    "rid",
    "tpl",
    "event",
    "wt_arr",
    "wt_dep",
    "wt_pass",
    "est",
    "actual",
    "message_time",
    "plat",
    "src",
]


def route_specs(routes: RouteSet, tiploc_crs: pd.DataFrame) -> list[ck.RouteSpec]:
    by_crs = tiploc_crs.dropna(subset=["crs"]).groupby("crs")["tiploc"].agg(frozenset)
    return [
        ck.RouteSpec(
            r.id,
            by_crs.get(r.origin, frozenset()),
            by_crs.get(r.destination, frozenset()),
        )
        for r in routes.routes
    ]


def _read(root: Path, table: str, ssd: date, columns: list[str] | None = None) -> pd.DataFrame:
    parts = sorted((root / table / f"service_date={ssd.isoformat()}").glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    return (
        ds.dataset([str(p) for p in parts], format="parquet").to_table(columns=columns).to_pandas()
    )


def process_day(root: Path, ssd: date, specs: list[ck.RouteSpec], tiploc_crs: pd.DataFrame) -> dict:
    sched = _read(root, "schedules", ssd)
    points = _read(root, "schedule_points", ssd)
    actuals = _read(root, "stop_actuals", ssd)
    if sched.empty:
        return {"checks": [], "route_cov": [], "toc_cov": [], "share": {"ssd": str(ssd)}}
    latest = ck.latest_schedules(sched)
    plat = ck.latest_points(points, latest)
    members = ck.route_membership(plat, specs)
    members["network"] = set(latest["rid"])
    crs = dict(zip(tiploc_crs["tiploc"], tiploc_crs["crs"], strict=True))
    rows: list[dict] = []

    def add(check: str, route: str, n_rows: int, n_flagged: int) -> None:
        rows.append(
            {
                "ssd": str(ssd),
                "check": check,
                "route": route,
                "n_rows": int(n_rows),
                "n_flagged": int(n_flagged),
            }
        )

    bad_act = (
        ck.schema_failures(actuals, ck.STOP_ACTUALS) if len(actuals) else pd.Series(dtype=bool)
    )
    if len(actuals):
        bad_act |= ck.utc_range_failures(actuals, ["actual"])
    bad_pts = ck.schema_failures(points, ck.SCHEDULE_POINTS)
    bad_pts |= ck.utc_range_failures(points, ["pt_arr", "pt_dep", "wt_arr", "wt_dep", "wt_pass"])
    bad_sch = ck.schema_failures(sched, ck.SCHEDULES)

    public = plat[plat["pt_arr"].notna() | plat["pt_dep"].notna()].copy()
    public["matched"] = public["tpl"].map(crs).notna()
    seq = ck.sequence_failures(actuals, plat) if len(actuals) else {}
    actual_rids = set(actuals["rid"]) if len(actuals) else set()
    classes = ck.closure_classes(latest, plat, actual_rids)
    # a service with no public calling time is not a passenger train, whatever its flag says
    classes.loc[~classes["rid"].isin(set(public["rid"])), "class"] = "not_passenger"
    no_sched = actual_rids - set(latest["rid"])
    dups = _duplicates(root, ssd)
    upd_rows = _update_rows(root, ssd)

    for route, rids in members.items():
        a_m = actuals["rid"].isin(rids) if len(actuals) else pd.Series(dtype=bool)
        p_m, s_m = points["rid"].isin(rids), sched["rid"].isin(rids)
        add(
            "types_utc",
            route,
            a_m.sum() + p_m.sum() + s_m.sum(),
            (bad_act[a_m].sum() if len(actuals) else 0) + bad_pts[p_m].sum() + bad_sch[s_m].sum(),
        )
        pm = public["rid"].isin(rids)
        add("station_matching", route, pm.sum(), (~public.loc[pm, "matched"]).sum())
        d_rids = [r for r in dups.index if r in rids] if len(dups) else []
        add(
            "duplicate_updates",
            route,
            sum(upd_rows.get(r, 0) for r in rids) if route != "network" else sum(upd_rows.values()),
            dups.loc[d_rids].sum() if d_rids else 0,
        )
        for name in (
            "impossible_arrival_before_prev_departure",
            "impossible_negative_dwell",
            "impossible_schedule_time_decreasing",
        ):
            s = seq.get(name)
            if s is None:
                add(name, route, 0, 0)
                continue
            mine = s[s.index.isin(rids)]
            add(name, route, len(mine), mine.sum())
        mine = classes[classes["rid"].isin(rids)]
        passenger = mine[mine["class"] != "not_passenger"]
        for cls in (
            "closure_cancellation_with_reason",
            "closure_cancellation_no_reason",
            "closure_missing_data",
            "closure_ran_with_actuals",
        ):
            add(cls, route, len(passenger), (passenger["class"] == cls).sum())
        add(
            "closure_not_planned_no_schedule",
            route,
            len(actual_rids & rids) if route != "network" else len(actual_rids),
            len(no_sched & rids) if route != "network" else len(no_sched),
        )

    month = str(ssd)[:7]
    route_cov = []
    for route, rids in members.items():
        mine = classes[classes["rid"].isin(rids) & (classes["class"] != "not_passenger")]
        route_cov.append(
            {
                "route": route,
                "month": month,
                "ssd": str(ssd),
                "scheduled": len(mine),
                "ran": int((mine["class"] == "closure_ran_with_actuals").sum()),
                "cancelled": int(mine["class"].str.startswith("closure_cancel").sum()),
            }
        )
    train_rids = set(classes.loc[classes["class"] != "not_passenger", "rid"])
    toc = latest[latest["rid"].isin(train_rids)].groupby("toc", dropna=False).size()
    toc_cov = [
        {"toc": str(k), "month": month, "ssd": str(ssd), "scheduled": int(v)}
        for k, v in toc.items()
    ]
    first_files = sched.groupby("rid")["src_file"].min().str.slice(0, 10)
    lead = (pd.to_datetime(str(ssd)) - pd.to_datetime(first_files)).dt.days
    share = {
        "ssd": str(ssd),
        "rids_with_updates": len(actual_rids | set(upd_rows)),
        "rids_with_schedule": len(set(upd_rows) & set(latest["rid"])),
        "schedule_first_in_earlier_file": int((lead > 0).sum()),
        "schedule_first_same_day_file": int((lead == 0).sum()),
        "schedule_first_next_day_file": int((lead < 0).sum()),
        "schedules": int(len(latest)),
    }
    return {"checks": rows, "route_cov": route_cov, "toc_cov": toc_cov, "share": share}


def _update_rows(root: Path, ssd: date) -> dict[str, int]:
    parts = sorted((root / "stop_updates" / f"service_date={ssd.isoformat()}").glob("*.parquet"))
    if not parts:
        return {}
    t = ds.dataset([str(p) for p in parts], format="parquet").to_table(columns=["rid"])
    counts = t.column("rid").value_counts().to_pylist()
    return {c["values"]: c["counts"] for c in counts}


def _duplicates(root: Path, ssd: date) -> pd.Series:
    parts = sorted((root / "stop_updates" / f"service_date={ssd.isoformat()}").glob("*.parquet"))
    if not parts:
        return pd.Series(dtype="int64")
    t = ds.dataset([str(p) for p in parts], format="parquet").to_table(columns=DUP_KEYS)
    return ck.duplicate_update_rids(t)


def summarise(per_day: list[dict]) -> pd.DataFrame:
    rows = [r for d in per_day for r in d["checks"]]
    df = pd.DataFrame(rows)
    return df.groupby(["check", "route"], as_index=False)[["n_rows", "n_flagged"]].sum()


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
