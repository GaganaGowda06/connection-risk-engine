import copy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rail_risk.ui import client
from rail_risk.ui.badges import badges

APP = str(Path(__file__).parent.parent / "src" / "rail_risk" / "ui" / "app.py")

ROUTES = [
    {"id": "bhm-eus", "origin": "BHM", "destination": "EUS", "model_trained_on_route": True},
    {"id": "ast-bhm", "origin": "AST", "destination": "BHM", "model_trained_on_route": False},
]
SERVICES = [
    {
        "rid": "202610158700587", "route_id": "bhm-eus", "toc": "VT", "origin": "BHM",
        "scheduled_departure": "19:54", "scheduled_arrival": "20:54", "minutes_to_arrival": 69,
        "started": True, "cancelled": False,
    },
    {
        "rid": "202610158980920", "route_id": "bhm-eus", "toc": "LM", "origin": "BHM",
        "scheduled_departure": "20:06", "scheduled_arrival": "21:17", "minutes_to_arrival": 92,
        "started": False, "cancelled": False,
    },
]  # fmt: skip
RISK = {
    "request_id": "r1", "rid": "202610158700587", "route_id": "bhm-eus", "status": "running",
    "toc": "VT", "scheduled_arrival": "20:54", "arrival_range_80": "20:53 to 21:14",
    "median_arrival": "21:00", "median_delay_minutes": 6.0, "p_delay_at_least_15": 0.085,
    "p_delay_at_least_30": 0.021, "p_delay_at_least_60": 0.003, "minutes_to_arrival_used": 69.0,
    "cancellation_reason": None,
    "delay_repay": {"toc": "VT", "toc_status": "verified", "amount_gbp": 1.39, "reason": None,
                    "caveat": None},
    "model_run_id": "ccaafb52a93b46e2a8c463b18efb8c42", "model_version": "1",
    "data_quality": {
        "live_data_age_minutes": 0.2, "stale_live_data": False, "model_fallback": False,
        "horizon_snapped_from_minutes": None, "horizon_used_minutes": 60,
        "route_in_training": True, "schedule_known": True, "features_age_seconds": 2.3,
        "notes": [],
    },
}  # fmt: skip


def risk(**changes):
    body = copy.deepcopy(RISK)
    for key, value in changes.items():
        if key == "dq":
            body["data_quality"].update(value)
        elif key == "repay":
            body["delay_repay"].update(value)
        else:
            body[key] = value
    return body


@pytest.fixture
def api(monkeypatch):
    """A mocked API: the page's only route to the model."""
    state = {"risk": [(200, risk(), 2)], "calls": [], "slept": []}
    monkeypatch.setattr(client, "get_routes", lambda: ROUTES)
    monkeypatch.setattr(client, "get_services", lambda route: SERVICES)

    def post_risk(rid, fare, ticket):
        state["calls"].append((rid, fare, ticket))
        queue = state["risk"]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(client, "post_risk", post_risk)
    monkeypatch.setattr(client, "sleep", lambda s: state["slept"].append(s))
    return state


def run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=20)
    at.run()
    assert not at.exception
    return at


def text_of(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def test_happy_path_shows_range_probabilities_and_payout(api):
    at = run()
    assert "# Expected 20:53 to 21:14" in text_of(at.markdown)
    labels = " ".join(str(p.proto.text) for p in at.get("progress"))
    assert "15 or more minutes late: 8%" in labels or "15 or more minutes late: 9%" in labels
    assert "30 or more minutes late: 2%" in labels and "60 or more minutes late: 0%" in labels
    assert [m.value for m in at.metric] == ["£1.39"]
    assert not at.warning and not at.error
    assert api["calls"][0] == ("202610158700587", None, "single")  # no fare entered
    assert any("How this works" in e.label for e in at.expander)


def test_fare_and_ticket_are_sent_to_the_api(api):
    at = run()
    at.number_input[0].set_value(50.0)
    at.run()
    assert api["calls"][-1][1] == 50.0
    at.selectbox[2].select("return")  # route, train, then ticket type
    at.run()
    assert api["calls"][-1][2] == "return"


@pytest.mark.parametrize(
    ("changes", "needle", "kind"),
    [
        (
            {"dq": {"stale_live_data": True, "live_data_age_minutes": 14.0}},
            "14 minutes old",
            "warning",
        ),
        ({"dq": {"model_fallback": True}}, "simple average of past trains", "warning"),
        (
            {"dq": {"horizon_snapped_from_minutes": 69.0, "horizon_used_minutes": 60}},
            "60 minute view",
            "info",
        ),
        ({"dq": {"route_in_training": False}}, "not trained on this route", "warning"),
        (
            {"repay": {"toc_status": "unverified", "amount_gbp": None, "reason": "unverified"}},
            "could not verify the Delay Repay rules for VT",
            "info",
        ),
    ],
)
def test_each_data_quality_badge_is_shown_with_a_plain_explanation(api, changes, needle, kind):
    api["risk"] = [(200, risk(**changes), 2)]
    at = run()
    shown = at.warning if kind == "warning" else at.info
    assert needle in text_of(shown)


def test_unverified_operator_shows_no_amount(api):
    api["risk"] = [(200, risk(repay={"toc_status": "unverified", "amount_gbp": None,
                                     "reason": "unverified"}), 2)]  # fmt: skip
    at = run()
    assert not at.metric
    assert "No amount" in text_of(at.caption) or "Enter your ticket price" in text_of(at.caption)


def test_warming_response_retries_once_then_shows_the_answer(api):
    api["risk"] = [(202, {"status": "warming"}, 3), (200, risk(), 2)]
    at = run()
    assert api["slept"] == [3]  # waited as told, once
    assert "# Expected 20:53 to 21:14" in text_of(at.markdown)
    assert len(api["calls"]) == 2


def test_warming_that_does_not_clear_shows_a_friendly_message(api):
    api["risk"] = [(202, {"status": "warming"}, 2)]
    at = run()
    assert len(api["calls"]) == 2  # one retry, no more
    assert "still gathering live data" in text_of(at.info)
    assert not at.error and not at.exception


def test_api_down_shows_a_plain_message_not_a_stack_trace(monkeypatch):
    def down():
        raise client.ApiDown("connection refused")

    monkeypatch.setattr(client, "get_routes", down)
    at = AppTest.from_file(APP, default_timeout=20)
    at.run()
    assert not at.exception
    assert "not reachable" in text_of(at.error)


def test_api_goes_down_mid_session(api, monkeypatch):
    def broken(rid, fare, ticket):
        raise client.ApiDown("timeout")

    monkeypatch.setattr(client, "post_risk", broken)
    at = run()
    assert "stopped answering" in text_of(at.error)


@pytest.mark.parametrize(
    ("status", "needle"),
    [(404, "no live information"), (504, "took too long"), (500, "code 500")],
)
def test_error_statuses_get_plain_messages(api, status, needle):
    api["risk"] = [(status, {}, 2)]
    at = run()
    assert needle in text_of(at.warning) + text_of(at.error)


def test_cancelled_and_arrived_services(api):
    api["risk"] = [(200, risk(status="cancelled", cancellation_reason="832",
                              arrival_range_80=None), 2)]  # fmt: skip
    at = run()
    assert "cancelled" in text_of(at.error) and "832" in text_of(at.error)
    api["risk"] = [(200, risk(status="arrived", median_arrival="21:03", median_delay_minutes=9.0,
                              arrival_range_80=None), 2)]  # fmt: skip
    at = run()
    assert "has arrived at 21:03" in text_of(at.success)


def test_no_services_message(monkeypatch, api):
    monkeypatch.setattr(client, "get_services", lambda route: [])
    at = run()
    assert "No trains are due" in text_of(at.info)


def test_badges_function_is_quiet_for_a_clean_answer():
    assert badges(RISK) == []


def test_risk_with_retry_waits_the_advertised_time_and_asks_twice(monkeypatch):
    calls, slept = [], []
    answers = iter([(202, {"status": "warming"}, 4), (200, {"status": "running"}, 2)])
    monkeypatch.setattr(client, "post_risk", lambda *a: calls.append(a) or next(answers))
    monkeypatch.setattr(client, "sleep", slept.append)
    assert client.risk_with_retry("1", None, "single") == (200, {"status": "running"})
    assert slept == [4] and len(calls) == 2


def test_the_page_does_not_import_the_model_layer():
    source = Path(APP).read_text()
    for forbidden in ("models.", "lightgbm", "mlflow", "registry", "RiskService", "pyspark"):
        assert forbidden not in source
