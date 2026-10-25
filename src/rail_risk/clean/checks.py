"""The Stage 1 cleaning checks: one pandera schema per table and the same checks on every route.

Nothing here knows a station, a route or an operator by name. Routes arrive as data (origin and
destination TIPLOC sets built from config/routes.yaml and the station map), so the live capture
on the Oracle host can run these functions unchanged on its own parsed tables.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
import pyarrow as pa_arrow
import pyarrow.compute as pc

SPLITS = ["train_pool", "winter_test", "oot_week"]
# Darwin train categories that are not trains: bus, bus replacement and ship. They carry public
# times but never train actuals, so they are not missing data.
NON_TRAIN_CATS = frozenset({"BR", "BS", "SS"})
RID = r"^\d{15}$"
DAY_US = 86_400 * 1_000_000
UTC = "datetime64[us, UTC]"

CHECKS = (
    "types_utc",
    "station_matching",
    "duplicate_updates",
    "impossible_arrival_before_prev_departure",
    "impossible_negative_dwell",
    "impossible_schedule_time_decreasing",
    "closure_cancellation_with_reason",
    "closure_cancellation_no_reason",
    "closure_missing_data",
    "closure_not_planned_no_schedule",
    "closure_ran_with_actuals",
)


def _ts(nullable: bool = True) -> pa.Column:
    return pa.Column(pd.DatetimeTZDtype(unit="us", tz="UTC"), nullable=nullable)


STOP_ACTUALS = pa.DataFrameSchema(
    {
        "rid": pa.Column(str, pa.Check.str_matches(RID)),
        "ssd": pa.Column(object),
        "tpl": pa.Column(str, pa.Check.str_length(1, 7)),
        "event": pa.Column(str, pa.Check.isin(["arr", "dep", "pass"])),
        "actual": _ts(False),
        "pt_arr": _ts(),
        "pt_dep": _ts(),
        "wt_arr": _ts(),
        "wt_dep": _ts(),
        "wt_pass": _ts(),
        "split": pa.Column(str, pa.Check.isin(SPLITS)),
    },
    coerce=False,
    strict=False,
)

SCHEDULE_POINTS = SCHEDULE_POINTS_COLUMNS = pa.DataFrameSchema(
    {
        "rid": pa.Column(str, pa.Check.str_matches(RID)),
        "ssd": pa.Column(object),
        "seq": pa.Column("int32", pa.Check.ge(0)),
        "kind": pa.Column(str, pa.Check.isin(["OR", "OPOR", "IP", "OPIP", "PP", "DT", "OPDT"])),
        "tpl": pa.Column(str, pa.Check.str_length(1, 7)),
        "cancelled": pa.Column(bool),
        "pt_arr": _ts(),
        "pt_dep": _ts(),
        "wt_arr": _ts(),
        "wt_dep": _ts(),
        "wt_pass": _ts(),
        "split": pa.Column(str, pa.Check.isin(SPLITS)),
    },
    strict=False,
)

SCHEDULES = pa.DataFrameSchema(
    {
        "rid": pa.Column(str, pa.Check.str_matches(RID)),
        "ssd": pa.Column(object),
        "toc": pa.Column(str, nullable=True),
        "is_passenger": pa.Column(bool),
        "train_cat": pa.Column(str, nullable=True),
        "is_active": pa.Column(bool),
        "deleted": pa.Column(bool),
        "message_time": _ts(False),
        "split": pa.Column(str, pa.Check.isin(SPLITS)),
    },
    strict=False,
)


@dataclass(frozen=True)
class RouteSpec:
    """A route as data: id plus the TIPLOCs that stand for its origin and destination."""

    id: str
    origin: frozenset[str]
    destination: frozenset[str]


def schema_failures(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.Series:
    """Number of failing rows per rid for a lazy validation (index is the row, value a flag)."""
    bad = pd.Series(False, index=df.index)
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        idx = exc.failure_cases["index"].dropna().astype(int).unique()
        bad.iloc[idx] = True
    return bad


def utc_range_failures(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    """Rows with any timestamp more than a day before or three days after the service date."""
    lo = pd.to_datetime(df["ssd"]).dt.tz_localize("UTC") - pd.Timedelta(days=1)
    hi = pd.to_datetime(df["ssd"]).dt.tz_localize("UTC") + pd.Timedelta(days=3)
    bad = pd.Series(False, index=df.index)
    for c in cols:
        bad |= (df[c] < lo) | (df[c] > hi)
    return bad


def route_membership(points: pd.DataFrame, routes: list[RouteSpec]) -> dict[str, set[str]]:
    """rids whose public calls reach a route's origin before its destination (any version)."""
    out: dict[str, set[str]] = {}
    calls = points[points["pt_arr"].notna() | points["pt_dep"].notna()]
    for route in routes:
        o = calls[calls["tpl"].isin(route.origin)].groupby("rid")["seq"].min()
        d = calls[calls["tpl"].isin(route.destination)].groupby("rid")["seq"].max()
        both = pd.concat([o.rename("o"), d.rename("d")], axis=1, join="inner")
        out[route.id] = set(both.index[both["o"] < both["d"]])
    return out


def latest_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    return schedules.sort_values("message_time").drop_duplicates("rid", keep="last")


def latest_points(points: pd.DataFrame, schedules_latest: pd.DataFrame) -> pd.DataFrame:
    """Calling points of each rid's newest schedule message only."""
    key = schedules_latest[["rid", "message_time"]]
    return points.merge(key, on=["rid", "message_time"], how="inner")


def closure_classes(
    schedules_latest: pd.DataFrame, points_latest: pd.DataFrame, actual_rids: set[str]
) -> pd.DataFrame:
    """Planned closure versus missing data, one class per rid.

    No schedule means not planned to run; a cancelled schedule with a reason is a cancellation; a
    schedule that is active, carries passengers, has no actuals and is not cancelled is missing
    data. A schedule that is deleted, inactive or has a cancelled point counts as cancelled.
    """
    cancelled_points = points_latest.groupby("rid")["cancelled"].any()
    s = schedules_latest.set_index("rid")
    cancelled = s["deleted"] | ~s["is_active"] | cancelled_points.reindex(s.index, fill_value=False)
    has_reason = s["cancel_reason"].notna()
    has_actual = pd.Series(s.index.isin(list(actual_rids)), index=s.index)
    cls = pd.Series("closure_missing_data", index=s.index)
    cls[has_actual] = "closure_ran_with_actuals"
    cls[cancelled & has_reason] = "closure_cancellation_with_reason"
    cls[cancelled & ~has_reason] = "closure_cancellation_no_reason"
    cls[~s["is_passenger"] | s["train_cat"].isin(NON_TRAIN_CATS)] = "not_passenger"
    return cls.rename("class").reset_index()


# Actual times are whole minutes (every one in the extract), and an arrival and a departure at one
# stop are separate events (often separate berth steps). An inversion of one reporting minute is
# therefore measurement resolution, not a train running backwards: it is counted apart and only a
# larger inversion is called impossible. The rule is the same on every route and stop.
RESOLUTION = pd.Timedelta(minutes=1)


def sequence_failures(actuals: pd.DataFrame, points: pd.DataFrame) -> dict[str, pd.Series]:
    """Impossible orderings, per rid: arrival before the previous departure, negative dwell, and
    scheduled working times that run backwards along a calling pattern. Inversions within the
    reporting resolution are returned separately as `*_within_resolution`."""
    out: dict[str, pd.Series] = {}
    a = actuals.copy()
    a["wt"] = a["wt_arr"].fillna(a["wt_dep"]).fillna(a["wt_pass"])
    keys = ["rid", "tpl", "wt_arr", "wt_dep"]
    arr = a[a["event"] == "arr"][[*keys, "wt", "actual"]].rename(columns={"actual": "arr"})
    dep = a[a["event"] == "dep"][[*keys, "wt", "actual"]].rename(columns={"actual": "dep"})
    stop = arr.merge(dep, on=keys, how="outer", suffixes=("", "_d"))
    stop["wt"] = stop["wt"].fillna(stop["wt_d"])
    stop = stop.sort_values(["rid", "wt"])
    gap = stop["arr"] - stop["dep"]
    neg = gap > RESOLUTION
    out["impossible_negative_dwell"] = neg.groupby(stop["rid"]).sum()
    out["resolution_negative_dwell"] = ((gap > pd.Timedelta(0)) & ~neg).groupby(stop["rid"]).sum()
    prev_dep = stop.groupby("rid")["dep"].shift(1)
    lead = prev_dep - stop["arr"]
    early = lead > RESOLUTION
    out["impossible_arrival_before_prev_departure"] = early.groupby(stop["rid"]).sum()
    out["resolution_arrival_before_prev_departure"] = (
        ((lead > pd.Timedelta(0)) & ~early).groupby(stop["rid"]).sum()
    )
    p = points.sort_values(["rid", "seq"]).copy()
    p["wt_any"] = p["wt_arr"].fillna(p["wt_pass"]).fillna(p["wt_dep"])
    back = p.groupby("rid")["wt_any"].diff() < pd.Timedelta(0)
    out["impossible_schedule_time_decreasing"] = back.groupby(p["rid"]).sum()
    return out


def duplicate_update_rids(updates: pa_arrow.Table) -> pd.Series:
    """Exact repeats of a stop update message (same rid, stop, event, content and message time)."""
    keys = [c for c in updates.column_names if c not in ("src_file", "split")]
    g = updates.group_by(keys, use_threads=True).aggregate([("rid", "count")])
    dup = g.filter(pc.greater(g["rid_count"], 1)).to_pandas()
    if dup.empty:
        return pd.Series(dtype="int64")
    return (dup["rid_count"] - 1).groupby(dup["rid"]).sum()
