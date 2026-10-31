# ruff: noqa: E501
"""Where every number the network service returns comes from (GOAL test 8).

Each response carries `provenance`: `fields` maps every numeric or time field to a source key and
the ledger column behind it, and `sources` says what each key is (run ids, the MCT file and its
RJTTF sequence, the message data clock, the guard file). The service prices nothing: no fare, no
Delay Repay amount is returned, so no Delay Repay rule is used here.

The ledger records, once per replay, the run ids that scored it and the MCT file its slack came
from (meta key `provenance`). The API compares them with the runs it names. A mismatch, or a
connection answer whose MCT source is unknown, is refused with a 503 rather than shown.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

from rail_risk.serving import artifacts as AR

MCT_TABLE = Path("data/ref/mct.csv")
MCT_RULE = "TSI time for the arriving and departing operator pair, else the station MSN time; no value is assumed"

# field -> (source key, ledger column or note)
HEALTH = {
    "data_as_of": ("message_data", "clock of the last committed replay hour"),
    "feed_age_min": ("message_data", "minutes from the last message to data_as_of"),
    "hours_done": ("ledger", "count of committed replay hours"),
    "stale": ("freshness_guard", "feed silence above flag_min"),
    "stale_reason": ("freshness_guard", "text of the silence"),
    "uptime_s": ("process", "seconds since the service started"),
    "status": ("freshness_guard", "degraded when stale"),
    "ids.arrival_model": ("run_ids", "serving/artifacts.py"),
    "ids.arrival_calibration": ("run_ids", "serving/artifacts.py"),
    "ids.pmiss_model": ("run_ids", "serving/artifacts.py"),
    "ids.adaptive_gamma": ("adaptive_gamma", "serving/artifacts.py"),
}
FRESH = {
    k: v for k, v in HEALTH.items() if k in ("data_as_of", "feed_age_min", "stale", "stale_reason")
}
FRESH["hours_done"] = HEALTH["hours_done"]
RISK_SNAP = {
    "horizon_min": (
        "snapshot_grid_arrival",
        "minutes before planned arrival the snapshot was taken",
    ),
    "scored_at": ("message_data", "snap_time: only messages stamped before it were used"),
    "route": ("timetable", "route of the service"),
    "point_min": ("arrival_model", "point"),
    "range80_min": ("arrival_calibration", "final_lo, final_hi"),
    "p_delay_15": ("arrival_calibration", "p_late_final"),
    "fallback_weight": ("freshness_guard", "fallback_w"),
    "stale": ("freshness_guard", "stale"),
    "stale_reason": ("freshness_guard", "stale_reason"),
    "sched_arr": ("timetable", "arrival.pt_arr: scheduled arrival at the route destination"),
}
CONN_SNAP = {
    "horizon_min": ("snapshot_grid_pair", "minutes before the outbound scheduled departure"),
    "scored_at": ("message_data", "snap_time: only messages stamped before it were used"),
    "hub": ("timetable", "hub of the interchange"),
    "p_miss": ("pmiss_model", "p_final"),
    "fallback_weight": ("freshness_guard", "fallback_w"),
    "slack_min": ("timetable_and_mct", "planned gap minus mct_min"),
    "mct_min": ("mct", "minutes from the MCT table for this station and operator pair"),
    "inbound_arrived": ("message_data", "inbound arrival message seen before scored_at"),
    "sched_arr_in": ("timetable", "pair.plan_arr_in: scheduled arrival of the inbound at the hub"),
    "sched_dep_out": (
        "timetable",
        "pair.plan_dep_out: scheduled departure of the outbound from the hub",
    ),
    "in_toc": ("timetable", "pair.in_toc: operator code of the inbound"),
    "out_toc": ("timetable", "pair.out_toc: operator code of the outbound"),
    "stale": ("freshness_guard", "stale"),
    "stale_reason": ("freshness_guard", "stale_reason"),
}
INBOUND_ROW = {
    "rid": ("request", "echo"),
    "planned_arrival": ("timetable", "plan_arr_in"),
    "connections_scored": ("ledger", "count of pair rows for this inbound"),
    "latest_p_miss": ("pmiss_model", "p_final of the latest snapshot"),
    "stale": ("freshness_guard", "stale"),
}
SERVICE_ROW = {
    "rid": ("ledger", "arrival.rid or pair.in_rid or pair.out_rid"),
    "toc": (
        "timetable",
        "pair.out_toc or pair.in_toc; null when the service is in no scored connection",
    ),
    "origin": ("timetable", "first part of arrival.route, or pair.hub for a departure from a hub"),
    "dest": ("timetable", "second part of arrival.route, or pair.hub for an arrival at a hub"),
    "sched_dep": (
        "timetable",
        "pair.plan_dep_out at the origin; null when no scored connection departs from it",
    ),
    "sched_arr": ("timetable", "arrival.pt_arr at the destination, or pair.plan_arr_in at a hub"),
    "has_risk": ("ledger", "the service has rows in the arrival table, so /risk answers"),
}
SERVICE_QUERY = {k: ("request", "echo of the request") for k in
                 ("origin", "dest", "near", "by", "window_min", "limit")}  # fmt: skip


def _expand(prefix: str, spec: dict) -> dict:
    return {f"{prefix}.{k}": v for k, v in spec.items()}


def fields_for(endpoint: str) -> dict:
    ids = {k: v for k, v in HEALTH.items() if k.startswith("ids.")}
    base = {**_expand("freshness", FRESH), **ids}
    if endpoint == "health":
        return dict(HEALTH)
    if endpoint == "risk":
        out = {"rid": ("request", "echo"), "toc": SERVICE_ROW["toc"], **base}
        out.update(_expand("snapshots[]", RISK_SNAP))
        out.update(_expand("latest", RISK_SNAP))
        return out
    if endpoint == "connection":
        out = {"in_rid": ("request", "echo"), "out_rid": ("request", "echo"), **base}
        out.update(_expand("snapshots[]", CONN_SNAP))
        out.update(_expand("latest", CONN_SNAP))
        return out
    if endpoint == "services":
        out = dict(base)
        out.update(_expand("query", SERVICE_QUERY))
        out.update(_expand("services[]", SERVICE_ROW))
        return out
    if endpoint == "inbound":
        out = {"hub": ("request", "echo"), **base}
        out.update(_expand("arrivals[]", INBOUND_ROW))
        return out
    raise KeyError(endpoint)


_TABLE_CACHE: dict = {}


def mct_from_table(path: Path | None = None) -> dict | None:
    """File, sequence and date of the MCT table on disk, or None when it is absent."""
    path = path or MCT_TABLE
    if not path.exists():
        return None
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE.clear()
        _TABLE_CACHE[key] = _read_table_head(path)
    return _TABLE_CACHE[key]


def _read_table_head(path: Path) -> dict:
    head = pd.read_csv(
        path, usecols=["source_file", "source_seq", "generated_date"]
    ).drop_duplicates()
    if len(head) != 1:
        return {"table": str(path), "problem": f"{len(head)} different source files in one table"}
    r = head.iloc[0]
    return {"table": str(path), "file": str(r["source_file"]), "sequence": int(r["source_seq"]),
            "generated": str(r["generated_date"]), "rule": MCT_RULE}  # fmt: skip


def runs_now() -> dict:
    return {"arrival_model": AR.ARRIVAL_RUN, "arrival_calibration": AR.CALIBRATION_RUN,
            "pmiss_model": AR.PMISS_RUN, "adaptive_gamma": AR.ADAPTIVE_GAMMA}  # fmt: skip


def stamp(mct: dict | None) -> dict:
    """What the replay writes into the ledger once: the runs that scored it and the MCT source."""
    return {"runs": runs_now(), "mct": mct}


def read_meta(con: sqlite3.Connection) -> dict | None:
    try:
        r = con.execute("select v from meta where k='provenance'").fetchone()
    except sqlite3.OperationalError:
        return None
    return json.loads(r[0]) if r else None


def check(stamped: dict | None, table: dict | None) -> dict:
    """Facts about the ledger the service can prove: do its recorded runs match the runs this
    service names, and where did its MCT values come from."""
    runs = (stamped or {}).get("runs")
    mct = (stamped or {}).get("mct") or table
    problems = []
    if runs is None:
        runs_match, note = None, "the ledger does not record the runs that scored it"
    else:
        bad = [k for k, v in runs_now().items() if runs.get(k) != v]
        runs_match = not bad
        note = "" if runs_match else "ledger scored by different runs: " + ", ".join(bad)
        if bad:
            problems.append(note)
    if not mct or "problem" in mct:
        problems_mct = (mct or {}).get(
            "problem"
        ) or "no MCT source: not in the ledger and no data/ref/mct.csv"
    else:
        problems_mct = ""
    return {"runs_match": runs_match, "runs_note": note, "mct": mct if not problems_mct else None,
            "mct_problem": problems_mct, "run_problems": problems}  # fmt: skip


def sources(endpoint: str, chk: dict, clock: str | None, guard_flag_min: float) -> dict:
    s = {
        "run_ids": {"note": "the three frozen MLflow runs this service names", **runs_now()},
        "arrival_model": {"run_id": AR.ARRIVAL_RUN, "output": "point"},
        "arrival_calibration": {"run_id": AR.CALIBRATION_RUN, "method": "CQR plus adaptive conformal inference", "gamma": AR.ADAPTIVE_GAMMA, "outputs": "final_lo, final_hi, p_late_final"},
        "pmiss_model": {"run_id": AR.PMISS_RUN, "output": "p_final"},
        "adaptive_gamma": {"value": AR.ADAPTIVE_GAMMA, "set_in": "serving/artifacts.py, chosen on validation (DECISIONS.md)"},
        "freshness_guard": {"file": "reports/serving/guard_v2.json", "flag_min": guard_flag_min},
        "message_data": {"as_of": clock, "feed": "Darwin messages, UTC; times are the ledger clock and snap_time"},
        "timetable": {"feed": "Darwin schedule messages"},
        "snapshot_grid_arrival": {"horizons_min": [15, 60], "set_in": "serving/engine.py"},
        "snapshot_grid_pair": {"horizons_min": [5, 15, 30, 60], "config": "config/connections.yaml"},
        "ledger": {"note": "rows committed by the replay engine"},
        "process": {"note": "this service's clock"},
        "request": {"note": "echoed from the request"},
        "ledger_runs": {"matches_service": chk["runs_match"], "note": chk["runs_note"]},
    }  # fmt: skip
    if endpoint in ("connection", "health"):
        s["mct"] = chk["mct"] or {"missing": chk["mct_problem"]}
        s["timetable_and_mct"] = {"timetable": "Darwin schedule messages", "mct": "see sources.mct"}
    return s


def block(endpoint: str, chk: dict, clock: str | None, guard_flag_min: float) -> dict:
    return {"fields": {k: f"{src}: {col}" for k, (src, col) in fields_for(endpoint).items()},
            "sources": sources(endpoint, chk, clock, guard_flag_min)}  # fmt: skip
