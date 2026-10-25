# ruff: noqa: E501
"""FastAPI service: /risk, /connection, /inbound, /health, all answered from the committed ledger.

Every response carries the model run ids, the data freshness (the clock of the last committed
hour and the age of the last message at that clock) and a stale flag with its reason. A request is
a lookup: the engine computes snapshots when their hour is ingested, the API never recomputes.
"""

import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rail_risk.logging import get_logger
from rail_risk.serving import artifacts as AR
from rail_risk.serving.engine import Thresholds
from rail_risk.serving.state import SqliteStore

log = get_logger("serving_api")
STATE_DIR = Path(os.environ.get("RAIL_SERVE_STATE", "data/interim/serve/state"))
THRESHOLDS = Path(os.environ.get("RAIL_GUARD_THRESHOLD", "reports/serving/guard_threshold.json"))
RIDPAT = r"^\d{15}$"
HUBS = Literal["BHM", "MAN", "BRI"]
STARTED = time.time()

app = FastAPI(title="Connection Risk Engine", version="6.0")


class RiskRequest(BaseModel):
    rid: str = Field(pattern=RIDPAT)


class ConnectionRequest(BaseModel):
    in_rid: str = Field(pattern=RIDPAT)
    out_rid: str = Field(pattern=RIDPAT)


def _store() -> SqliteStore:
    return SqliteStore(STATE_DIR / "state.sqlite")


def _meta(store: SqliteStore) -> dict:
    blob = store.load_blob()
    th = Thresholds.load(THRESHOLDS)
    if not blob:
        return {"data_as_of": None, "feed_age_min": None, "stale": True,
                "stale_reason": "no data ingested yet", "hours_done": 0}  # fmt: skip
    clock = pd.Timestamp(blob["clock"])
    minutes = np.array(blob["minutes"], dtype="int64")
    now_min = int(clock.timestamp() // 60)
    last = minutes[minutes <= now_min]
    age = float(now_min - last.max()) if len(last) else 1e6
    stale = age > th.feed_min
    return {
        "data_as_of": str(clock), "feed_age_min": round(age, 1), "stale": bool(stale),
        "stale_reason": f"no message for {age:.0f} min" if stale else "", "hours_done": None,
    }  # fmt: skip


def _ids() -> dict:
    return {"arrival_model": AR.ARRIVAL_RUN, "arrival_calibration": AR.CALIBRATION_RUN,
            "pmiss_model": AR.PMISS_RUN, "adaptive_gamma": AR.ADAPTIVE_GAMMA}  # fmt: skip


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


@app.get("/health")
def health() -> dict:
    store = _store()
    try:
        meta = _meta(store)
        meta["hours_done"] = len(store.hours_done())
        meta["ids"] = _ids()
        meta["uptime_s"] = round(time.time() - STARTED, 1)
        meta["status"] = "degraded" if meta["stale"] else "ok"
        return meta
    finally:
        store.close()


@app.post("/risk")
def risk(req: RiskRequest) -> dict:
    t0 = time.perf_counter()
    store = _store()
    try:
        rows = store.risk(req.rid)
        meta = _meta(store)
    finally:
        store.close()
    if rows.empty:
        raise HTTPException(404, "no snapshot for this service yet")
    rows = rows.sort_values(["snap_time"])
    snaps = []
    for r in rows.itertuples():
        snaps.append({
            "horizon_min": int(r.horizon_min), "scored_at": r.snap_time, "route": r.route,
            "point_min": _clean(r.point), "range80_min": [_clean(r.final_lo), _clean(r.final_hi)],
            "p_delay_15": _clean(r.p_late_final), "stale": bool(r.stale),
            "stale_reason": r.stale_reason or "",
        })  # fmt: skip
    out = {
        "rid": req.rid,
        "snapshots": snaps,
        "latest": snaps[-1],
        "freshness": meta,
        "ids": _ids(),
    }
    log.info(
        "risk", rid=req.rid, ms=round((time.perf_counter() - t0) * 1000, 2), stale=meta["stale"]
    )
    return out


@app.post("/connection")
def connection(req: ConnectionRequest) -> dict:
    t0 = time.perf_counter()
    store = _store()
    try:
        rows = store.pair(req.in_rid, req.out_rid)
        meta = _meta(store)
    finally:
        store.close()
    if rows.empty:
        raise HTTPException(404, "no scored connection for this pair yet")
    rows = rows.sort_values("snap_time")
    snaps = [
        {"horizon_min": int(r.horizon_min), "scored_at": r.snap_time, "hub": r.hub,
         "p_miss": _clean(r.p_final), "slack_min": _clean(r.slack_min), "mct_min": _clean(r.mct_min),
         "inbound_arrived": bool(not r.nya), "stale": bool(r.stale), "stale_reason": r.stale_reason or ""}
        for r in rows.itertuples()
    ]  # fmt: skip
    out = {"in_rid": req.in_rid, "out_rid": req.out_rid, "snapshots": snaps, "latest": snaps[-1],
           "freshness": meta, "ids": _ids()}  # fmt: skip
    log.info("connection", ms=round((time.perf_counter() - t0) * 1000, 2), stale=meta["stale"])
    return out


@app.get("/inbound")
def inbound(hub: HUBS, limit: int = Query(50, ge=1, le=500)) -> dict:
    store = _store()
    try:
        meta = _meta(store)
        if not meta["data_as_of"]:
            raise HTTPException(503, "no data yet")
        clock = pd.Timestamp(meta["data_as_of"])
        start = (clock - pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
        end = (clock + pd.Timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S")
        rows = store.inbound(hub, start, end)
    finally:
        store.close()
    if rows.empty:
        return {"hub": hub, "arrivals": [], "freshness": meta, "ids": _ids()}
    g = (
        rows.sort_values("snap_time")
        .groupby("in_rid")
        .tail(1)
        .sort_values("plan_arr_in")
        .head(limit)
    )
    arr = [{"rid": r.in_rid, "planned_arrival": r.plan_arr_in, "connections_scored": int(
        (rows["in_rid"] == r.in_rid).sum()), "latest_p_miss": _clean(r.p_final), "stale": bool(r.stale)}
        for r in g.itertuples()]  # fmt: skip
    return {"hub": hub, "arrivals": arr, "freshness": meta, "ids": _ids()}


def worker_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}
