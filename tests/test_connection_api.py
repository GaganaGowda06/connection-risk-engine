"""POST /connection and GET /inbound on a small synthetic Darwin world.

Times are written in UTC; 15 October 2026 is British Summer Time, so the clock times the API
returns and accepts are one hour later (the helper `local` converts).
"""

import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rail_risk.alerts import load_alert_config
from rail_risk.api.app import create_app
from rail_risk.api.schemas import ConnectionResponse
from rail_risk.api.service import RiskService
from rail_risk.config import load_routes
from rail_risk.connection import FixedDelays, load_stations
from rail_risk.live import LiveStore
from rail_risk.repay import load_rules
from world import BANDS, TODAY, ts

TIPLOCS = {"BHAMNWS": "BHM", "EUSTON": "EUS", "ASTON": "AST", "MNCRPIC": "MAN"}
INBOUND = "202610150000101"
NOW = ts("10:25").to_pydatetime()  # 11:25 local; the inbound is due at 10:37 UTC
RANGE = re.compile(r"^\d\d:\d\d to \d\d:\d\d$")
COLUMNS = [
    "rid", "ssd", "tiploc", "event_type", "time_kind", "time", "planned_time", "late_reason",
    "message_ts", "partition", "offset",
]  # fmt: skip


def local(hhmm_utc: str) -> str:
    h, m = hhmm_utc.split(":")
    return f"{int(h) + 1:02d}:{m}"


def stops(rid: str, items, first_offset: int = 0) -> list[dict]:
    """items: (tiploc, 'arr' or 'dep', planned, actual or None) in UTC."""
    rows = []
    for i, (tpl, etype, planned, actual) in enumerate(items):
        kind, when = ("actual", actual) if actual else ("forecast", planned)
        rows.append(
            {
                "rid": rid, "ssd": pd.Timestamp(TODAY), "tiploc": tpl, "event_type": etype,
                "time_kind": kind, "time": ts(when), "planned_time": ts(planned),
                "late_reason": None, "message_ts": ts("10:20"), "partition": 0,
                "offset": first_offset + i,
            }
        )  # fmt: skip
    return rows


def world(inbound_actual: str | None = None, started: bool = True, extra=()):
    """The inbound AST 10:30 to BHM 10:37, and outbounds BHM to EUS at 10:42, 10:47, 10:55,
    11:05 and 11:20 (UTC), each taking 80 minutes."""
    rows = stops(
        INBOUND,
        [
            ("ASTON", "dep", "10:30", "10:31" if started else None),
            ("BHAMNWS", "arr", "10:37", inbound_actual),
        ],
        0,
    )
    outbound = {}
    for k, dep in enumerate(["10:42", "10:47", "10:55", "11:05", "11:20"]):
        rid = f"2026101500002{k:02d}"
        h, m = int(dep[:2]), int(dep[3:])
        arr = f"{h + (m + 80) // 60:02d}:{(m + 80) % 60:02d}"
        outbound[dep] = rid
        rows += stops(
            rid, [("BHAMNWS", "dep", dep, None), ("EUSTON", "arr", arr, None)], 10 + 4 * k
        )
    for r in extra:
        rows += r
    return pd.DataFrame(rows, columns=COLUMNS), outbound


def make(tmp_path, features, bundle_pair, *, events=None, newest="10:24", now=NOW, warm=True,
         fallback=False, cancel=(), history=True, tau=0.2):  # fmt: skip
    bundle, base = bundle_pair
    ev, _ = events if events is not None else world()
    store = LiveStore(tmp_path / "interim", tmp_path / "raw", TIPLOCS)
    store.events = ev
    for rid in set(ev["rid"]):
        store.schedules[rid] = {
            "rid": rid, "toc": "VT", "cancel_reason": "100" if rid in cancel else None,
            "n_stops": 2, "n_cancelled_stops": 0, "message_ts": ts("10:00"), "offset": 0,
        }  # fmt: skip
    store.newest_message_ts = ts(newest).to_pydatetime()
    service = RiskService(
        store=store, features=features, routes=load_routes("config/routes.yaml").routes,
        bands=BANDS, rules=load_rules("config/delay_repay.yaml"),
        bundle=None if fallback else bundle, baseline=base, model_run_id="run-test",
        model_version="7", route_model_trained_on={"bhm-eus", "ast-bhm"}, now=lambda: now,
        outbound_history=FixedDelays(2.0, 3.0) if history else None,
        stations=load_stations("config/stations.yaml"), alert_tau=tau,
    )  # fmt: skip
    if warm:
        service.refresh_features()
    return service


def client(service) -> TestClient:
    return TestClient(create_app(lambda: service, refresh_seconds=0))


def connect(c, **body):
    return c.post("/connection", json={"inbound_rid": INBOUND, **body})


def test_feasible_connection_returns_the_full_answer(tmp_path, features, bundle_pair):
    with client(make(tmp_path, features, bundle_pair)) as c:
        r = connect(c, outbound_departure=local("10:47"), fare=50, ticket="single")
        assert r.status_code == 200
        body = r.json()
        ConnectionResponse.model_validate(body)
        assert body["status"] == "ok" and body["inbound"]["state"] == "approaching"
        out = body["outbound"]
        assert out["scheduled_departure"] == local("10:47") and out["chosen_automatically"] is False
        assert out["scheduled_gap_minutes"] == 10.0 and len(out["alternatives"]) == 3
        assert 0.0 <= body["p_miss"] <= 1.0 and 0.0 <= body["p_stranded"] <= body["p_miss"]
        assert RANGE.match(body["arrival_range_80"]) and re.match(
            r"^\d\d:\d\d$", body["median_arrival"]
        )
        ct = body["connection_time"]
        assert (ct["station"], ct["minutes"], ct["status"]) == ("BHM", 10, "assumed")
        rows = {s["minutes"]: s for s in body["sensitivity"]}
        assert list(rows) == [5, 7, 10, 15]
        assert rows[15]["feasible"] is False and rows[15]["p_miss"] is None  # gap 10 < 15
        assert rows[5]["p_miss"] <= rows[7]["p_miss"] <= rows[10]["p_miss"]  # monotone in time
        assert rows[10]["p_miss"] == body["p_miss"]
        dr = body["delay_repay"]
        assert dr["toc_status"] == "verified" and dr["amount_gbp"] is not None
        assert "stranded" in dr["stranded_assumption"]
        text = " ".join(body["assumptions"]).lower()
        assert "independent" in text and "assumed" in text
        assert body["model_run_id"] == "run-test"
        dq = body["data_quality"]
        assert dq["stale_live_data"] is False and dq["model_fallback"] is False
        assert body["alert_threshold"] == 0.2
        assert body["alert"] is (body["p_miss"] >= 0.2)


def test_no_outbound_given_chooses_the_first_feasible(tmp_path, features, bundle_pair):
    with client(make(tmp_path, features, bundle_pair)) as c:
        body = connect(c).json()
        assert body["status"] == "ok" and body["outbound"]["chosen_automatically"] is True
        # due 10:37, so 10:42 is 5 minutes (too tight) and 10:47 is the first with 10
        assert body["outbound"]["scheduled_departure"] == local("10:47")


def test_infeasible_outbound_is_refused_naming_the_first_feasible(tmp_path, features, bundle_pair):
    with client(make(tmp_path, features, bundle_pair)) as c:
        r = connect(c, outbound_departure=local("10:42"))
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "refused" and body["outbound"] is None
        assert "infeasible on paper" in body["refusal"] and local("10:47") in body["refusal"]
        assert body["first_feasible_departure"] == local("10:47")
        assert body["p_miss"] is None and body["alert"] is None and body["sensitivity"] == []
        assert body["connection_time"]["status"] == "assumed"  # still stated


def test_unlisted_departure_is_refused(tmp_path, features, bundle_pair):
    with client(make(tmp_path, features, bundle_pair)) as c:
        body = connect(c, outbound_departure="23:59").json()
        assert body["status"] == "refused" and "no planned outbound" in body["refusal"]


def test_a_very_late_inbound_is_stranded_and_the_range_is_withheld(tmp_path, features, bundle_pair):
    # arrived at 12:37 UTC, two hours late, after every outbound listed has gone
    ev = world(inbound_actual="12:37")
    service = make(
        tmp_path, features, bundle_pair, events=ev, newest="12:30", now=ts("12:40").to_pydatetime()
    )
    with client(service) as c:
        body = connect(c, outbound_departure=local("10:47")).json()
        assert body["status"] == "ok" and body["inbound"]["state"] == "arrived"
        assert body["inbound"]["expected_delay_minutes"] == 120.0
        assert body["p_miss"] == 1.0 and body["p_stranded"] == 1.0
        assert body["arrival_range_80"] is None and body["median_arrival"] is None
        assert body["alert"] is True
        assert body["delay_repay"]["amount_gbp"] is None or body["delay_repay"]["amount_gbp"] >= 0


def test_an_inbound_that_has_arrived_on_time_makes_the_connection(tmp_path, features, bundle_pair):
    ev = world(inbound_actual="10:37")
    with client(make(tmp_path, features, bundle_pair, events=ev, newest="10:36",
                     now=ts("10:38").to_pydatetime())) as c:  # fmt: skip
        body = connect(c, outbound_departure=local("10:47")).json()
        assert body["inbound"]["state"] == "arrived" and body["p_miss"] == 0.0
        assert body["alert"] is False


def test_stale_data_treats_the_inbound_as_not_started_and_says_so(tmp_path, features, bundle_pair):
    service = make(tmp_path, features, bundle_pair, newest="10:05")
    with client(service) as c:
        body = connect(c, outbound_departure=local("10:47")).json()
        dq = body["data_quality"]
        assert body["status"] == "ok" and dq["stale_live_data"] is True
        assert any("older than" in n for n in dq["notes"])


def test_alert_fires_at_the_threshold_and_not_just_below_it(tmp_path, features, bundle_pair):
    service = make(tmp_path, features, bundle_pair)
    with client(service) as c:
        p = connect(c, outbound_departure=local("10:47")).json()["p_miss"]
        assert 0.0 <= p < 1.0
        for tau, expected in ((p, True), (p + 0.0001, False), (max(p - 0.0001, 0.0), True)):
            service.alert_tau = tau
            body = connect(c, outbound_departure=local("10:47")).json()
            assert body["p_miss"] == p  # the same draws every time
            assert body["alert"] is expected, (tau, p)
            assert body["alert_threshold"] == tau


def test_cancelled_inbound_and_cancelled_outbound(tmp_path, features, bundle_pair):
    with client(make(tmp_path, features, bundle_pair, cancel={INBOUND})) as c:
        body = connect(c).json()
        assert body["status"] == "cancelled" and body["inbound"]["state"] == "cancelled"
        assert "cancelled" in body["refusal"] and body["p_miss"] is None
    ev = world()
    cancelled = ev[1]["10:47"]
    with client(make(tmp_path, features, bundle_pair, events=ev, cancel={cancelled})) as c:
        refused = connect(c, outbound_departure=local("10:47")).json()
        assert refused["status"] == "refused" and "cancelled" in refused["refusal"]
        assert refused["first_feasible_departure"] == local("10:55")
        auto = connect(c).json()  # a cancelled train is never offered
        assert auto["outbound"]["scheduled_departure"] == local("10:55")


def test_outbound_that_has_already_left_is_refused(tmp_path, features, bundle_pair):
    ev, outbound = world()
    left = ev.copy()
    mask = (left["rid"] == outbound["10:47"]) & (left["event_type"] == "dep")
    left.loc[mask, ["time_kind", "time"]] = ["actual", ts("10:49")]
    with client(make(tmp_path, features, bundle_pair, events=(left, outbound))) as c:
        body = connect(c, outbound_departure=local("10:47")).json()
        assert body["status"] == "refused" and "already left" in body["refusal"]


def test_model_fallback_is_visible_and_still_answers(tmp_path, features, bundle_pair):
    with client(make(tmp_path, features, bundle_pair, fallback=True)) as c:
        body = connect(c, outbound_departure=local("10:47")).json()
        dq = body["data_quality"]
        assert body["status"] == "ok" and dq["model_fallback"] is True
        assert any("model unavailable" in n for n in dq["notes"]) and body["model_run_id"] is None


def test_errors_and_validation(tmp_path, features, bundle_pair):
    with client(make(tmp_path, features, bundle_pair)) as c:
        assert c.post("/connection", json={"inbound_rid": "202610159999999"}).status_code == 404
        ev, outbound = world()
        not_inbound = c.post("/connection", json={"inbound_rid": outbound["10:47"]})
        assert not_inbound.status_code == 404
        assert "Aston to New Street" in not_inbound.json()["detail"]
        for bad in (
            {"inbound_rid": "12"},
            {"inbound_rid": INBOUND, "outbound_departure": "25:00"},
            {"inbound_rid": INBOUND, "fare": 5000},
            {"inbound_rid": INBOUND, "ticket": "season"},
            {"inbound_rid": INBOUND, "extra": 1},
        ):
            assert c.post("/connection", json=bad).status_code == 422, bad
    with client(make(tmp_path, features, bundle_pair, history=False)) as c:
        assert connect(c).status_code == 503


def test_cold_cache_returns_202_warming_for_the_inbound(tmp_path, features, bundle_pair):
    service = make(tmp_path, features, bundle_pair, warm=False)
    with client(service) as c:
        r = connect(c, outbound_departure=local("10:47"))
        assert r.status_code == 202 and r.headers["retry-after"] == "2"
        assert r.json()["status"] == "warming" and service._queue[0] == INBOUND
        service.refresh_features()
        assert connect(c, outbound_departure=local("10:47")).status_code == 200


def test_inbound_lists_only_aston_services_due_within_two_hours(tmp_path, features, bundle_pair):
    with client(make(tmp_path, features, bundle_pair)) as c:
        items = c.get("/inbound").json()
        assert [i["rid"] for i in items] == [INBOUND]
        assert items[0]["route_id"] == "ast-bhm" and items[0]["scheduled_arrival"] == local("10:37")
        assert items[0]["minutes_to_arrival"] == 12 and items[0]["cancelled"] is False


def test_alert_threshold_is_read_from_the_config_file():
    cfg = load_alert_config("config/alerts.yaml")
    assert cfg.tau == pytest.approx(0.22) and cfg.false_alert == 1.0 and cfg.missed_alert == 5.0
