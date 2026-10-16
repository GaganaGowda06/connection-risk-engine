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


INBOUND = [
    {
        "rid": "202610168740001", "route_id": "ast-bhm", "toc": "LM", "origin": "AST",
        "scheduled_departure": "10:00", "scheduled_arrival": "10:07", "minutes_to_arrival": 12,
        "started": True, "cancelled": False,
    },
]  # fmt: skip
CONNECTION = {
    "request_id": "c1", "status": "ok",
    "inbound": {"rid": "202610168740001", "scheduled_arrival": "10:07", "state": "approaching",
                "minutes_to_arrival": 12, "expected_delay_minutes": 2.0},
    "outbound": {"rid": "202610168740200", "toc": "VT", "scheduled_departure": "10:17",
                 "scheduled_arrival": "11:37", "scheduled_gap_minutes": 10.0,
                 "chosen_automatically": True, "alternatives": ["10:30", "10:45", "11:00"]},
    "refusal": None, "first_feasible_departure": None, "p_miss": 0.2, "alert": False,
    "alert_threshold": 0.22, "p_stranded": 0.0, "arrival_range_80": "11:37 to 12:05",
    "median_arrival": "11:45",
    "delay_repay": {"toc": "VT", "toc_status": "verified", "amount_gbp": 2.4, "reason": None,
                    "caveat": None,
                    "stranded_assumption": "A draw with no catchable train is priced in the top."},
    "connection_time": {"station": "BHM", "minutes": 10, "status": "assumed", "comment": None},
    "sensitivity": [
        {"minutes": 5, "feasible": True, "p_miss": 0.08, "arrival_range_80": "11:37 to 12:00",
         "expected_payout_gbp": 1.9},
        {"minutes": 7, "feasible": True, "p_miss": 0.12, "arrival_range_80": "11:37 to 12:02",
         "expected_payout_gbp": 2.0},
        {"minutes": 10, "feasible": True, "p_miss": 0.2, "arrival_range_80": "11:37 to 12:05",
         "expected_payout_gbp": 2.4},
        {"minutes": 15, "feasible": False, "p_miss": None, "arrival_range_80": None,
         "expected_payout_gbp": None},
    ],
    "assumptions": ["The legs are modelled as independent.", "The connection time is assumed."],
    "model_run_id": "4c88d55bbbed49018e3ce05e8199ce5b", "model_version": "2",
    "data_quality": {
        "live_data_age_minutes": 0.3, "stale_live_data": False, "model_fallback": False,
        "horizon_snapped_from_minutes": None, "horizon_used_minutes": 12,
        "route_in_training": True, "schedule_known": True, "features_age_seconds": 3.0,
        "notes": [],
    },
}  # fmt: skip


def connection(**changes):
    body = copy.deepcopy(CONNECTION)
    for key, value in changes.items():
        if key == "dq":
            body["data_quality"].update(value)
        else:
            body[key] = value
    return body


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
    state["inbound"] = []
    state["connection"] = [(200, connection(), 2)]
    monkeypatch.setattr(client, "get_inbound", lambda: state["inbound"])

    def post_connection(rid, departure, fare, ticket):
        state["conn_calls"].append((rid, departure, fare, ticket))
        queue = state["connection"]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    state["conn_calls"] = []
    monkeypatch.setattr(client, "post_connection", post_connection)
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


# the My connection tab


def conn_run(api, **body) -> AppTest:
    api["inbound"] = INBOUND
    api["connection"] = [(200, connection(**body), 2)]
    return run()


def test_connection_tab_says_the_chance_in_plain_words_and_shows_euston_range(api):
    at = conn_run(api)
    assert [t.label for t in at.tabs] == ["Train arrival", "My connection"]
    text = text_of(at.markdown)
    assert "About 1 in 5 chance you miss the 10:17." in text
    assert "# Expected 11:37 to 12:05" in text
    assert "£2.40" in [m.value for m in at.metric]
    assert not at.error  # 0.20 is under the 0.22 alert level
    assert api["conn_calls"][0] == ("202610168740001", None, None, "single")
    shown = text_of(at.caption) + text
    assert "first one you can make" in shown and "stranded" not in shown.lower()


def test_alert_banner_appears_only_when_the_api_says_so(api):
    at = conn_run(api, alert=True, p_miss=0.3)
    banner = text_of(at.error)
    assert "Your connection is at risk" in banner and "30%" in banner and "22%" in banner
    assert "About 1 in 3 chance" in text_of(at.markdown)
    assert not conn_run(api, alert=False).error


def test_sensitivity_and_assumptions_are_in_the_expander(api):
    at = conn_run(api)
    labels = [e.label for e in at.expander]
    assert "Connection time sensitivity and assumptions" in labels
    table = at.table[0].value
    assert list(table["Change time (minutes)"]) == [5, 7, 10, 15]
    assert list(table["Chance you miss it"]) == ["8%", "12%", "20%", "not possible"]
    body = text_of(at.markdown)
    assert "assumed" in body and "The legs are modelled as independent." in body


def test_refused_connection_names_the_first_feasible_train(api):
    at = conn_run(
        api, status="refused", outbound=None, p_miss=None, alert=None, sensitivity=[],
        refusal="infeasible on paper: the 10:12 leaves 5 min after the inbound is due",
        first_feasible_departure="10:17",
    )  # fmt: skip
    assert "infeasible on paper" in text_of(at.warning)
    assert "leaves New Street at 10:17" in text_of(at.info)


def test_cancelled_inbound_is_an_error_message(api):
    at = conn_run(
        api, status="cancelled", outbound=None, refusal="the inbound service is cancelled"
    )
    assert "cancelled" in text_of(at.error)


def test_no_arrival_time_when_stranded(api):
    at = conn_run(api, arrival_range_80=None, median_arrival=None, p_miss=1.0, p_stranded=1.0,
                  alert=True)  # fmt: skip
    assert "no train was left" in text_of(at.warning)
    assert "almost certainly miss the 10:17" in text_of(at.markdown)
    assert "100% chance that no later train" in text_of(at.caption)


def test_data_quality_badges_show_on_the_connection_tab(api):
    at = conn_run(api, dq={"stale_live_data": True, "live_data_age_minutes": 14.0})
    assert "Live data is old" in text_of(at.warning)


def test_connection_warming_retries_once_then_a_friendly_message(api):
    api["inbound"] = INBOUND
    api["connection"] = [(202, {"status": "warming"}, 3), (202, {"status": "warming"}, 3)]
    at = run()
    assert len(api["conn_calls"]) == 2 and api["slept"] == [3]
    assert "still gathering live data" in text_of(at.info)
    assert [b.label for b in at.button] == ["Try again"]


def test_connection_wanted_time_is_validated_and_sent(api):
    api["inbound"] = INBOUND
    at = run()
    at.text_input[0].set_value("10:2x")
    at.run()
    assert "for example 10:06" in text_of(at.warning)
    n = len(api["conn_calls"])
    at.text_input[0].set_value("10:30")
    at.run()
    assert api["conn_calls"][-1][1] == "10:30" and len(api["conn_calls"]) == n + 1


@pytest.mark.parametrize("status", [404, 503, 504])
def test_connection_error_codes_are_friendly(api, status):
    api["inbound"] = INBOUND
    api["connection"] = [(status, {"detail": "x"}, 2)]
    at = run()
    assert not at.exception and not at.error


def test_connection_tab_with_no_inbound_and_with_the_api_down(api, monkeypatch):
    at = run()
    assert "No trains from Aston" in text_of(at.info)

    def down():
        raise client.ApiDown("refused")

    monkeypatch.setattr(client, "get_inbound", down)
    at = run()
    assert "Could not load the list of trains" in text_of(at.error)
    assert "Expected 20:53 to 21:14" in text_of(at.markdown)  # the first tab still works


def test_connection_page_text_has_no_em_dash(api):
    at = conn_run(api, alert=True)
    shown = " ".join(
        str(e.value)
        for group in (at.markdown, at.caption, at.error, at.info, at.warning)
        for e in group
    )
    assert "\u2014" not in shown
    source = open(APP, encoding="utf-8").read()
    assert "\u2014" not in source


def test_words_for_chances():
    from rail_risk.ui.words import chance_in_words, miss_sentence

    assert chance_in_words(0.2) == "about 1 in 5" and chance_in_words(0.5) == (
        "more likely than not (about 50%)"
    )
    assert chance_in_words(0.004) == "less than 1 in 100" and chance_in_words(0.0) == "no chance"
    assert chance_in_words(0.95) == "almost certain"
    assert chance_in_words(0.41) == "about 4 in 10" and chance_in_words(0.3) == "about 1 in 3"
    assert miss_sentence(0.0, "10:06") == "You are very unlikely to miss the 10:06."
    assert miss_sentence(0.2, "10:06") == "About 1 in 5 chance you miss the 10:06."
    assert (
        miss_sentence(0.7, "10:06")
        == "It is more likely than not (about 70%) that you miss the 10:06."
    )
