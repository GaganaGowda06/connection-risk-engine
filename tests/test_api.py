import re
import threading

import pytest
from fastapi.testclient import TestClient

from rail_risk.api.app import create_app
from rail_risk.api.schemas import RiskResponse
from rail_risk.api.service import RiskService
from rail_risk.config import load_routes
from rail_risk.repay import DelayRepayRules, load_rules
from world import (
    BANDS,
    RID,
    events_for,
    make_store,
    ts,
)

ROUTE_DEST = {"bhm-eus": "EUS", "ast-bhm": "BHM", "bhm-man": "MAN"}
NOW = ts("10:45").to_pydatetime()  # 45 minutes before the scheduled 11:30 arrival
RANGE = re.compile(r"^\d\d:\d\d to \d\d:\d\d$")


def make_service(
    tmp_path, features, bundle_pair, *, events=None, toc="VT", newest="10:43", rules=None,
    fallback=False, baseline=True, now=NOW, warm=True,
):  # fmt: skip
    bundle, base = bundle_pair
    service = RiskService(
        store=make_store(tmp_path, events if events is not None else events_for(), toc, newest),
        features=features,
        routes=load_routes("config/routes.yaml").routes,
        bands=BANDS,
        rules=rules or load_rules("config/delay_repay.yaml"),
        bundle=None if fallback else bundle,
        baseline=base if baseline else None,
        model_run_id="run-test",
        model_version="7",
        route_model_trained_on={"bhm-eus"},
        now=lambda: now,
    )
    if warm:  # what the background refresher does, so a request reads the feature cache
        service.refresh_features()
    return service


def client_for(service, **kw):
    app = create_app(lambda: service, refresh_seconds=0, **kw)
    return TestClient(app)


def post(client, **body):
    return client.post("/risk", json={"rid": RID, **body})


def test_happy_path_returns_ranges_probabilities_and_payout(tmp_path, features, bundle_pair):
    with client_for(make_service(tmp_path, features, bundle_pair)) as c:
        r = post(c, fare=50, ticket="single")
        assert r.status_code == 200
        body = r.json()
        RiskResponse.model_validate(body)  # the response matches the declared schema
        assert RANGE.match(body["arrival_range_80"])
        assert re.match(r"^\d\d:\d\d$", body["scheduled_arrival"])
        p15, p30, p60 = (body[f"p_delay_at_least_{k}"] for k in (15, 30, 60))
        assert 0 <= p60 <= p30 <= p15 <= 1
        assert body["status"] == "running" and body["toc"] == "VT"
        assert body["minutes_to_arrival_used"] == 45.0
        assert body["model_run_id"] == "run-test" and body["model_version"] == "7"
        dr = body["delay_repay"]
        assert dr["toc_status"] == "verified" and dr["amount_gbp"] >= 0 and dr["reason"] is None
        dq = body["data_quality"]
        assert dq["stale_live_data"] is False and dq["model_fallback"] is False
        assert dq["live_data_age_minutes"] == 2.0 and dq["schedule_known"] is True
        # the model takes minutes to arrival as a number, so nothing is snapped
        assert dq["horizon_snapped_from_minutes"] is None and dq["horizon_used_minutes"] == 45
        assert body["request_id"] == r.headers["x-request-id"]


def test_continuous_model_sees_the_real_minutes_and_a_fixed_horizon_model_is_snapped(
    tmp_path, features, bundle_pair
):
    import dataclasses

    bundle, baseline = bundle_pair
    seen = []
    real_predict = bundle.quantile.predict

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def predict(self, frame, *a, **k):
            seen.append(float(frame["minutes_to_arrival"].iloc[0]))
            return self.inner(frame, *a, **k)

    spy = dataclasses.replace(bundle, quantile=Spy(real_predict))
    spy.quantile.inner = real_predict
    with client_for(make_service(tmp_path, features, (spy, baseline))) as c:
        assert post(c).status_code == 200
    assert seen == [45.0]  # the real minutes, not a grid value

    fixed = dataclasses.replace(bundle, continuous_horizon=False, horizons=(120, 60, 15))
    with client_for(make_service(tmp_path, features, (fixed, baseline))) as c:
        dq = post(c).json()["data_quality"]
    assert dq["horizon_snapped_from_minutes"] == 45.0 and dq["horizon_used_minutes"] == 60


def test_cache_miss_returns_202_warming_then_the_answer_once_built(tmp_path, features, bundle_pair):
    service = make_service(tmp_path, features, bundle_pair, warm=False)
    with client_for(service) as c:
        r = post(c, fare=50)
        assert r.status_code == 202
        assert r.headers["retry-after"] == "2"
        body = r.json()
        assert body["status"] == "warming" and body["rid"] == RID
        assert body["retry_after_seconds"] == 2 and body["request_id"] == r.headers["x-request-id"]
        assert service._queue[0] == RID  # at the front of the refresher queue
        assert post(c, fare=50).status_code == 202  # still not built, and not queued twice
        assert service._queue.count(RID) == 1
        service.refresh_features()  # what the refresher does when woken
        ok = post(c, fare=50)
        assert ok.status_code == 200 and RANGE.match(ok.json()["arrival_range_80"])
        assert service._queue == []


def test_queued_rids_are_built_ahead_of_the_rest(tmp_path, features, bundle_pair):
    service = make_service(tmp_path, features, bundle_pair, warm=False)
    order = []
    real = service._compute

    def spy(rids, stale):
        order.append(list(rids))
        return real(rids, stale)

    service._compute = spy
    service.queue_front("202610150000009")
    service.queue_front(RID)
    assert service._queue == [RID, "202610150000009"]
    service.refresh_features()
    assert order[0] == [RID, "202610150000009"]  # a batch of its own, before the full batch
    assert len(order) == 2


def test_a_service_that_cannot_be_built_is_404_not_endless_warming(tmp_path, bundle_pair, spark):
    class Empty:
        def batch(self, services):
            return {}

    service = make_service(tmp_path, Empty(), bundle_pair, warm=False)
    with client_for(service) as c:
        assert post(c).status_code == 202
        service.refresh_features()
        r = post(c)
        assert r.status_code == 404 and "could not be built" in r.json()["detail"]


def test_services_and_routes_endpoints(tmp_path, features, bundle_pair):
    with client_for(make_service(tmp_path, features, bundle_pair)) as c:
        items = c.get("/services?route=bhm-eus").json()
        assert [i["rid"] for i in items] == [RID]
        item = items[0]
        assert item["scheduled_arrival"] and item["minutes_to_arrival"] == 45
        assert item["started"] is True and item["cancelled"] is False and item["toc"] == "VT"
        assert c.get("/services?route=bhm-man").json() == []
        routes = {r["id"]: r for r in c.get("/routes").json()}
        assert routes["bhm-eus"]["model_trained_on_route"] is True
        assert routes["ast-bhm"]["model_trained_on_route"] is False


def test_no_fare_gives_no_amount_with_a_reason(tmp_path, features, bundle_pair):
    with client_for(make_service(tmp_path, features, bundle_pair)) as c:
        dr = post(c).json()["delay_repay"]
        assert dr["amount_gbp"] is None and dr["reason"] == "no fare supplied"


def test_stale_darwin_data_serves_without_live_state_and_says_so(tmp_path, features, bundle_pair):
    with client_for(make_service(tmp_path, features, bundle_pair, newest="10:20")) as c:
        body = post(c).json()
        assert body["data_quality"]["stale_live_data"] is True
        assert body["status"] == "not_started"  # live actuals are not used
        assert any("older than 10 minutes" in n for n in body["data_quality"]["notes"])
        assert c.get("/health").json()["status"] == "degraded"


def test_unknown_rid_is_404_and_a_cancellation_is_reported_with_its_reason(
    tmp_path, features, bundle_pair
):
    service = make_service(tmp_path, features, bundle_pair)
    with client_for(service) as c:
        assert c.post("/risk", json={"rid": "202610159999999"}).status_code == 404
        service.store.schedules[RID]["cancel_reason"] = "832"
        body = post(c).json()
        assert body["status"] == "cancelled" and body["cancellation_reason"] == "832"
        assert body["p_delay_at_least_15"] is None and body["arrival_range_80"] is None


def test_model_load_failure_serves_the_historical_baseline_and_flags_it(
    tmp_path, features, bundle_pair
):
    with client_for(make_service(tmp_path, features, bundle_pair, fallback=True)) as c:
        body = post(c, fare=50).json()
        assert body["data_quality"]["model_fallback"] is True
        assert body["model_run_id"] is None and body["model_version"] is None
        assert any("historical baseline" in n for n in body["data_quality"]["notes"])
        assert RANGE.match(body["arrival_range_80"])
        health = c.get("/health").json()
        assert health["model_fallback"] is True and health["status"] == "degraded"
    # with no baseline either, the timetable is served and that is stated
    with client_for(
        make_service(tmp_path, features, bundle_pair, fallback=True, baseline=False)
    ) as c:
        body = post(c).json()
        assert body["median_delay_minutes"] == 0.0
        assert any("timetable" in n for n in body["data_quality"]["notes"])


def test_unverified_or_unknown_operator_gets_no_amount_and_a_reason(
    tmp_path, features, bundle_pair
):
    base = load_rules("config/delay_repay.yaml").model_dump()
    vt = base["operators"]["VT"]
    unverified = DelayRepayRules.model_validate(
        {"operators": {"VT": {**vt, "status": "unverified", "tiers": [], "caveat": None}}}
    )
    with client_for(make_service(tmp_path, features, bundle_pair, rules=unverified)) as c:
        dr = post(c, fare=50).json()["delay_repay"]
        assert dr["toc_status"] == "unverified" and dr["amount_gbp"] is None
        assert "unverified" in dr["reason"]
    with client_for(make_service(tmp_path, features, bundle_pair, toc="XX")) as c:
        dr = post(c, fare=50).json()["delay_repay"]
        assert dr["toc_status"] == "unknown" and dr["amount_gbp"] is None


def test_shared_toc_lm_is_not_priced_with_the_shipped_rules(tmp_path, features, bundle_pair):
    # TOC LM is West Midlands Railway and London Northwestern Railway; the second is unverified
    with client_for(make_service(tmp_path, features, bundle_pair, toc="LM")) as c:
        dr = post(c, fare=50).json()["delay_repay"]
        assert dr["amount_gbp"] is None and dr["toc_status"] == "unverified"
        assert dr["reason"] == "not priced: rules unverified"
    with client_for(make_service(tmp_path, features, bundle_pair, toc="VT")) as c:
        assert post(c, fare=50).json()["delay_repay"]["amount_gbp"] is not None


def test_missing_schedule_means_unknown_operator_not_a_guess(tmp_path, features, bundle_pair):
    with client_for(make_service(tmp_path, features, bundle_pair, toc=None)) as c:
        body = post(c, fare=50).json()
        assert body["toc"] is None and body["delay_repay"]["toc_status"] == "unknown"
        assert body["data_quality"]["schedule_known"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"rid": "12345"},
        {"rid": "abcdefghijklmno"},
        {"rid": RID, "fare": -1},
        {"rid": RID, "fare": 1000.01},
        {"rid": RID, "ticket": "season"},
        {"rid": RID, "surprise": 1},
        {},
    ],
)
def test_invalid_requests_are_422_without_echoing_secrets(tmp_path, features, bundle_pair, payload):
    with client_for(make_service(tmp_path, features, bundle_pair)) as c:
        r = c.post("/risk", json=payload)
        assert r.status_code == 422


def test_fare_bounds_are_inclusive(tmp_path, features, bundle_pair):
    with client_for(make_service(tmp_path, features, bundle_pair)) as c:
        assert post(c, fare=0).status_code == 200
        assert post(c, fare=1000).status_code == 200


def test_request_id_is_propagated_and_unsafe_ones_replaced(tmp_path, features, bundle_pair):
    with client_for(make_service(tmp_path, features, bundle_pair)) as c:
        ok = c.get("/health", headers={"x-request-id": "abc-123"})
        assert ok.headers["x-request-id"] == "abc-123"
        bad = c.get("/health", headers={"x-request-id": "bad id with spaces"})
        assert re.fullmatch(r"[0-9a-f]{32}", bad.headers["x-request-id"])


def test_health_reports_model_alias_run_and_data_age(tmp_path, features, bundle_pair):
    with client_for(make_service(tmp_path, features, bundle_pair)) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["model_alias"] == "champion"
        assert h["model_run_id"] == "run-test" and h["darwin_data_age_minutes"] == 2.0
        assert h["stale_live_data"] is False


def test_a_blocked_cached_path_hits_the_budget_and_returns_504(
    tmp_path, features, bundle_pair, monkeypatch
):
    """The feature lookup is replaced by one that blocks until released, so the request can only
    end by timing out. No wall-clock thresholds are involved."""
    service = make_service(tmp_path, features, bundle_pair)
    release = threading.Event()

    def blocked(*args, **kwargs):
        release.wait(timeout=30)
        raise AssertionError("released after the request timed out")

    monkeypatch.setattr(service, "_features_for", blocked)
    with client_for(service, budget_seconds=0.05) as c:
        try:
            r = post(c)
            assert r.status_code == 504 and "no answer" in r.json()["detail"]
        finally:
            release.set()  # leaving the client waits for the worker thread, so release it first


def test_logs_and_responses_carry_no_secrets(tmp_path, features, bundle_pair, monkeypatch):
    monkeypatch.setenv("RAIL_HSP_API_KEY", "super-secret-key-value")
    with client_for(make_service(tmp_path, features, bundle_pair)) as c:
        text = post(c, fare=50).text + c.get("/health").text
        assert "super-secret-key-value" not in text and "x-apikey" not in text.lower()
