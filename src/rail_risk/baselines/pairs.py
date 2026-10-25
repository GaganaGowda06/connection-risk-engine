"""Interchange pairs for the gap-bucket baseline and test 5.

A pair is an inbound service (the interchange's inbound origin to the hub, one operator) and an
outbound service on the interchange's outbound route, on the same service date, with the outbound
planned departure from the hub at or after the inbound planned arrival. Plans come from the first
schedule message seen for each rid. Actuals are the last reported arrival at the hub (inbound) and
departure from the hub (outbound). An outbound with no actual at all anywhere is data missing, so
the pair is dropped; an outbound with actuals but none at the hub never left it, which is a miss.
Nothing here names a hub, a route or an operator: all of it comes from config/routes.yaml.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from rail_risk.baselines.services import day_services
from rail_risk.clean import checks as ck
from rail_risk.config import Interchange, RouteSet
from rail_risk.ingest.pport_extract import train_pool_table

PAIRS_DIR = "pairs"
MAX_GAP_MIN = 180.0
MAX_ACTUAL_GAP = pd.Timedelta(hours=12)
PAIR_COLS = [
    "ssd", "hub", "outbound", "in_rid", "out_rid", "in_toc", "out_toc", "plan_arr_in",
    "plan_dep_out", "act_arr_in", "act_dep_out", "in_arr_msg_time", "in_seen", "out_seen",
]  # fmt: skip


def inbound_id(ix: Interchange) -> str:
    return f"in:{ix.hub}:{ix.inbound_origin}-{ix.inbound_destination}:{ix.inbound_toc}"


def interchange_specs(
    routes: RouteSet, ix: Interchange, tiploc_crs: pd.DataFrame
) -> list[ck.RouteSpec]:
    """Route specs for the inbound leg and the outbound route of one interchange."""
    by_crs = tiploc_crs.dropna(subset=["crs"]).groupby("crs")["tiploc"].agg(frozenset)
    out = next(r for r in routes.routes if r.id == ix.outbound)
    empty = frozenset()
    return [
        ck.RouteSpec(
            inbound_id(ix),
            by_crs.get(ix.inbound_origin, empty),
            by_crs.get(ix.inbound_destination, empty),
        ),
        ck.RouteSpec(out.id, by_crs.get(out.origin, empty), by_crs.get(out.destination, empty)),
    ]


def _actuals(root: Path, ssd: date) -> pd.DataFrame:
    cols = ["rid", "tpl", "event", "actual", "first_actual_message_time"]
    a = train_pool_table(root, "stop_actuals", cols, [ssd])
    return a.to_pandas() if a.num_rows else pd.DataFrame(columns=cols)


def _nearest(p: pd.DataFrame, actuals: pd.DataFrame, rid: str, tpl: str, plan: str, col: str):
    """Attach the actual nearest the plan (the feed can carry a second row a day later); drop
    candidates more than 12 hours from the plan. Pairs with no usable actual keep NaT."""
    p = p.reset_index(drop=True).assign(_i=lambda x: x.index)
    m = p[["_i", rid, tpl, plan]].merge(
        actuals, left_on=[rid, tpl], right_on=["rid", "tpl"], how="inner"
    )
    m = m[(m[col] - m[plan]).abs() <= MAX_ACTUAL_GAP]
    m = m.assign(_gap=(m[col] - m[plan]).abs()).sort_values(["_i", "_gap"])
    m = m.drop_duplicates("_i")
    extra = [c for c in actuals.columns if c not in ("rid", "tpl")]
    return p.merge(m[["_i", *extra]], on="_i", how="left").drop(columns="_i")


def build_pairs_day(
    root: Path, ssd: date, routes: RouteSet, tiploc_crs: pd.DataFrame
) -> pd.DataFrame:
    actuals = _actuals(root, ssd)
    if actuals.empty:
        return pd.DataFrame(columns=PAIR_COLS)
    ran = set(actuals.loc[actuals["actual"].notna(), "rid"])
    arr = actuals[(actuals["event"] == "arr") & actuals["actual"].notna()]
    arr = arr.rename(
        columns={"actual": "act_arr_in", "first_actual_message_time": "in_arr_msg_time"}
    )[["rid", "tpl", "act_arr_in", "in_arr_msg_time"]]
    dep = actuals[(actuals["event"] == "dep") & actuals["actual"].notna()]
    dep = dep.rename(columns={"actual": "act_dep_out"})[["rid", "tpl", "act_dep_out"]]
    frames = []
    for ix in routes.interchanges:
        in_spec, out_spec = interchange_specs(routes, ix, tiploc_crs)
        services = day_services(root, ssd, [in_spec, out_spec])
        if services.empty:
            continue
        inb = services[(services["route"] == in_spec.id) & (services["toc"] == ix.inbound_toc)]
        out = services[services["route"] == out_spec.id]
        if inb.empty or out.empty:
            continue
        p = inb[["rid", "toc", "dest_tpl", "pt_arr", "schedule_seen"]].merge(
            out[["rid", "toc", "origin_tpl", "pt_dep_origin", "schedule_seen"]],
            how="cross",
            suffixes=("_in", "_out"),
        )
        gap = (p["pt_dep_origin"] - p["pt_arr"]).dt.total_seconds() / 60.0
        p = p[(gap >= 0) & (gap <= MAX_GAP_MIN)]
        p = p[p["rid_out"].isin(ran)]
        p = _nearest(p, arr, "rid_in", "dest_tpl", "pt_arr", "act_arr_in")
        p = _nearest(p, dep, "rid_out", "origin_tpl", "pt_dep_origin", "act_dep_out")
        p = p.rename(
            columns={
                "rid_in": "in_rid",
                "rid_out": "out_rid",
                "toc_in": "in_toc",
                "toc_out": "out_toc",
                "pt_arr": "plan_arr_in",
                "pt_dep_origin": "plan_dep_out",
                "schedule_seen_in": "in_seen",
                "schedule_seen_out": "out_seen",
            }
        )
        p["ssd"], p["hub"], p["outbound"] = ssd, ix.hub, ix.outbound
        frames.append(p[PAIR_COLS])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIR_COLS)


def known_at(pairs: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """Add forecast time T = planned inbound arrival minus the horizon and the two state flags."""
    out = pairs.copy()
    t = out["plan_arr_in"] - pd.to_timedelta(horizon_min, unit="m")
    out["horizon_min"] = horizon_min
    out["schedule_known"] = (out["in_seen"] <= t) & (out["out_seen"] <= t)
    out["inbound_arrived_by_t"] = out["in_arr_msg_time"].notna() & (out["in_arr_msg_time"] <= t)
    return out


def build_pairs_range(
    root: Path,
    days: list[date],
    routes: RouteSet,
    tiploc_crs: pd.DataFrame,
    out_dir: Path,
    rebuild: bool = False,
) -> dict:
    report: dict = {"built": 0, "skipped_existing": 0, "pairs": 0}
    for d in days:
        dest = out_dir / f"service_date={d}" / "part.parquet"
        if dest.exists() and not rebuild:
            report["skipped_existing"] += 1
            continue
        df = build_pairs_day(root, d, routes, tiploc_crs)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        report["built"] += 1
        report["pairs"] += len(df)
    return report


def load_pairs(out_dir: Path, days: list[date]) -> pd.DataFrame:
    parts = [out_dir / f"service_date={d}" / "part.parquet" for d in days]
    frames = [pd.read_parquet(p) for p in parts if p.exists()]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIR_COLS)
