# ruff: noqa: E501
"""FastAPI service: /risk, /connection, /inbound, /health, all answered from the committed ledger.

Every response carries the model run ids, the data freshness (the clock of the last committed
hour and the age of the last message at that clock) and a stale flag with its reason. A request is
a lookup: the engine computes snapshots when their hour is ingested, the API never recomputes.
"""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rail_risk.logging import get_logger
from rail_risk.serving import artifacts as AR
from rail_risk.serving import provenance as PV
from rail_risk.serving.guard import Guard
from rail_risk.serving.state import SqliteStore

log = get_logger("serving_api")
STATE_DIR = Path(os.environ.get("RAIL_SERVE_STATE", "data/interim/serve/state"))
RIDPAT = r"^\d{15}$"
HUBS = Literal["BHM", "MAN", "BRI"]
STARTED = time.time()

app = FastAPI(title="Connection Risk Engine", version="6.0")


class RiskRequest(BaseModel):
    rid: str = Field(pattern=RIDPAT)


class ConnectionRequest(BaseModel):
    in_rid: str = Field(pattern=RIDPAT)
    out_rid: str = Field(pattern=RIDPAT)


_LOCAL = threading.local()


def _store() -> SqliteStore:
    """One read-only connection per request thread and ledger, opened once and kept (a new
    connection per request was the cost that made the single-process service collapse at 150
    requests a second)."""
    path = STATE_DIR / "state.sqlite"
    cache = _LOCAL.__dict__.setdefault("stores", {})
    s = cache.get(path)
    if s is None:
        if not path.exists():  # nothing replayed yet: an empty ledger, so /health can say so
            SqliteStore(path).close()
        try:
            s = SqliteStore(path, readonly=True, keep_open=True)
        except sqlite3.OperationalError as e:
            raise HTTPException(503, f"state not ready: {e}") from e
        cache[path] = s
    return s


_FRESH_CACHE: dict = {}


def _fresh(store: SqliteStore) -> tuple[pd.Timestamp, int | None] | None:
    f = store.load_fresh()
    if f is None:  # a state written before the small summary existed: read the blob once per clock
        blob = store.load_blob()
        if not blob:
            return None
        key = blob["clock"]
        if key not in _FRESH_CACHE:
            minutes = np.array(blob["minutes"], dtype="int64")
            now_min = int(pd.Timestamp(key).timestamp() // 60)
            last = minutes[minutes <= now_min]
            _FRESH_CACHE.clear()
            _FRESH_CACHE[key] = int(last.max()) if len(last) else None
        return pd.Timestamp(key), _FRESH_CACHE[key]
    return pd.Timestamp(f["clock"]), f["last_minute"]


def _meta(store: SqliteStore) -> dict:
    flag_min = _guard().flag_min
    got = _fresh(store)
    if not got:
        return {"data_as_of": None, "feed_age_min": None, "stale": True,
                "stale_reason": "no data ingested yet", "hours_done": 0}  # fmt: skip
    clock, last = got
    age = float(int(clock.timestamp() // 60) - last) if last is not None else 1e6
    stale = age > flag_min
    return {
        "data_as_of": str(clock), "feed_age_min": round(age, 1), "stale": bool(stale),
        "stale_reason": f"no message for {age:.0f} min" if stale else "", "hours_done": None,
    }  # fmt: skip


_TH: dict = {}


def _guard() -> Guard:
    if "g" not in _TH:
        path = Path(os.environ.get("RAIL_GUARD", "reports/serving/guard_v2.json"))
        _TH["g"] = Guard.from_json(json.loads(path.read_text())) if path.exists() else Guard()
    return _TH["g"]


def _ids() -> dict:
    return {"arrival_model": AR.ARRIVAL_RUN, "arrival_calibration": AR.CALIBRATION_RUN,
            "pmiss_model": AR.PMISS_RUN, "adaptive_gamma": AR.ADAPTIVE_GAMMA}  # fmt: skip


def _chk(store: SqliteStore, need_mct: bool = False) -> dict:
    """The ledger's provenance facts; refuses (503) when its runs differ from the ones this
    service names, or, for a connection answer, when the MCT source is unknown."""
    chk = PV.check(PV.read_meta(store.con), PV.mct_from_table())
    if chk["run_problems"]:
        raise HTTPException(503, "; ".join(chk["run_problems"]))
    if need_mct and chk["mct_problem"]:
        raise HTTPException(503, f"mct_min has no source: {chk['mct_problem']}")
    return chk


def _prov(endpoint: str, chk: dict, meta: dict) -> dict:
    return PV.block(endpoint, chk, meta.get("data_as_of"), _guard().flag_min)


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
        meta["hours_done"] = store.hours_count()
        meta["ids"] = _ids()
        meta["uptime_s"] = round(time.time() - STARTED, 1)
        chk = PV.check(PV.read_meta(store.con), PV.mct_from_table())
        problems = [*chk["run_problems"], *([chk["mct_problem"]] if chk["mct_problem"] else [])]
        meta["status"] = "degraded" if meta["stale"] or problems else "ok"
        meta["problems"] = problems
        meta["provenance"] = _prov("health", chk, meta)
        meta["provenance"]["checks"] = {"ledger_runs_match_service": chk["runs_match"],
                                        "ledger_runs_note": chk["runs_note"]}  # fmt: skip
        return meta
    finally:
        store.close()


@app.post("/risk")
def risk(req: RiskRequest) -> dict:
    t0 = time.perf_counter()
    store = _store()
    try:
        chk = _chk(store)
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
            "fallback_weight": _clean(getattr(r, "fallback_w", 0.0)),
            "stale_reason": r.stale_reason or "",
        })  # fmt: skip
    out = {
        "rid": req.rid,
        "snapshots": snaps,
        "latest": snaps[-1],
        "freshness": meta,
        "ids": _ids(),
        "provenance": _prov("risk", chk, meta),
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
        chk = _chk(store, need_mct=True)
        rows = store.pair(req.in_rid, req.out_rid)
        meta = _meta(store)
    finally:
        store.close()
    if rows.empty:
        raise HTTPException(404, "no scored connection for this pair yet")
    rows = rows.sort_values("snap_time")
    snaps = [
        {"horizon_min": int(r.horizon_min), "scored_at": r.snap_time, "hub": r.hub,
         "p_miss": _clean(r.p_final), "fallback_weight": _clean(getattr(r, "fallback_w", 0.0)), "slack_min": _clean(r.slack_min), "mct_min": _clean(r.mct_min),
         "inbound_arrived": bool(not r.nya), "stale": bool(r.stale), "stale_reason": r.stale_reason or ""}
        for r in rows.itertuples()
    ]  # fmt: skip
    out = {"in_rid": req.in_rid, "out_rid": req.out_rid, "snapshots": snaps, "latest": snaps[-1],
           "freshness": meta, "ids": _ids(), "provenance": _prov("connection", chk, meta)}  # fmt: skip
    log.info("connection", ms=round((time.perf_counter() - t0) * 1000, 2), stale=meta["stale"])
    return out


@app.get("/inbound")
def inbound(
    hub: HUBS,
    limit: int = Query(50, ge=1, le=500),
    at: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}$"),
) -> dict:
    """Services due at the hub in the next hour. `at` (UTC, replay only) asks as of an earlier
    clock; the default is the ledger's clock."""
    store = _store()
    try:
        chk = _chk(store)
        meta = _meta(store)
        if not meta["data_as_of"]:
            raise HTTPException(503, "no data yet")
        clock = pd.Timestamp(at, tz="UTC") if at else pd.Timestamp(meta["data_as_of"])
        start = (clock - pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
        end = (clock + pd.Timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S")
        rows = store.inbound_summary(hub, start, end, limit)
    except sqlite3.OperationalError as e:
        raise HTTPException(503, f"state not ready: {e}") from e
    if rows.empty:
        return {"hub": hub, "arrivals": [], "freshness": meta, "ids": _ids(),
                "provenance": _prov("inbound", chk, meta)}  # fmt: skip
    arr = [
        {"rid": r.in_rid, "planned_arrival": r.plan_arr_in, "connections_scored": int(r.n),
         "latest_p_miss": _clean(r.p_final), "stale": bool(r.stale)}
        for r in rows.itertuples()
    ]  # fmt: skip
    return {"hub": hub, "arrivals": arr, "freshness": meta, "ids": _ids(),
            "provenance": _prov("inbound", chk, meta)}  # fmt: skip


def worker_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}
