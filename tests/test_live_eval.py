"""Point-in-time replay and capture-gap exclusion for the live evaluation."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk import live_eval as le
from rail_risk.config import load_routes
from rail_risk.live import LiveStore
from rail_risk.repay import load_rules
from world import BANDS, EVENT_COLUMNS, RID, TIPLOCS, TODAY, ts

# capture gaps


def ns(*stamps: pd.Timestamp) -> np.ndarray:
    return np.array([s.value for s in stamps], dtype="int64")


def minutes(start: str, end: str, skip: tuple[str, str] | None = None) -> np.ndarray:
    """One message a minute from start to end (UTC clock times), none strictly inside `skip`."""
    out = []
    for t in pd.date_range(ts(start), ts(end), freq="1min"):
        if skip and ts(skip[0]) < t < ts(skip[1]):
            continue
        out.append(t)
    return ns(*out)


def test_gap_longer_than_five_minutes_is_found_and_a_shorter_one_is_not():
    stamps = np.concatenate(
        [minutes("09:00", "09:20"), minutes("09:27", "09:40"), minutes("09:44", "10:00")]
    )  # 7 minute hole (09:20 to 09:27) and a 4 minute hole (09:40 to 09:44)
    gaps = le.find_gaps(stamps, ts("09:00"), ts("10:00"))
    assert len(gaps) == 1
    assert gaps.iloc[0]["gap_start"] == ts("09:20") and gaps.iloc[0]["gap_end"] == ts("09:27")
    assert gaps.iloc[0]["minutes"] == 7.0


def test_a_gap_of_exactly_five_minutes_is_not_a_gap():
    stamps = np.concatenate([minutes("09:00", "09:10"), minutes("09:15", "09:30")])
    assert le.find_gaps(stamps, ts("09:00"), ts("09:30")).empty


def test_gap_is_clipped_to_the_window_and_edges_count():
    stamps = ns(ts("08:50"), ts("09:30"))  # no message between, so the whole window is a gap
    gaps = le.find_gaps(stamps, ts("09:00"), ts("09:20"))
    assert len(gaps) == 1
    assert gaps.iloc[0]["gap_start"] == ts("09:00") and gaps.iloc[0]["gap_end"] == ts("09:20")
    assert (
        le.find_gaps(np.empty(0, dtype="int64"), ts("09:00"), ts("09:20")).iloc[0]["minutes"] == 20
    )


def test_a_message_just_outside_the_window_decides_the_edge():
    stamps = np.concatenate([ns(ts("08:58")), minutes("09:03", "09:30")])
    gaps = le.find_gaps(stamps, ts("09:00"), ts("09:30"))  # 08:58 to 09:03 is five minutes
    assert gaps.empty
    stamps = np.concatenate([ns(ts("08:50")), minutes("09:03", "09:30")])
    gaps = le.find_gaps(stamps, ts("09:00"), ts("09:30"))
    assert len(gaps) == 1 and gaps.iloc[0]["gap_start"] == ts("09:00")


def test_gap_hours_are_split_at_local_midnight():
    gaps = pd.DataFrame(
        {
            "gap_start": [pd.Timestamp("2026-10-17 22:30", tz="UTC")],  # 23:30 in London
            "gap_end": [pd.Timestamp("2026-10-17 23:30", tz="UTC")],  # 00:30 in London
            "minutes": [60.0],
        }
    )
    days = [date(2026, 10, 17), date(2026, 10, 18)]
    hours = le.gap_hours_by_day(gaps, days)
    assert hours == {days[0]: 0.5, days[1]: 0.5}
    assert le.gap_hours_by_day(gaps, days, (5, 24)) == {days[0]: 0.5, days[1]: 0.0}


def test_overlap_counts_shared_time_and_ignores_touching():
    gi = le.GapIndex(
        pd.DataFrame({"gap_start": [ts("10:40")], "gap_end": [ts("10:47")], "minutes": [7.0]})
    )
    assert gi.overlaps(ts("10:30"), ts("11:30"))  # the gap is inside the window
    assert gi.overlaps(ts("10:45"), ts("11:30"))  # the window starts inside the gap
    assert not gi.overlaps(ts("10:47"), ts("11:30"))  # the window starts as the gap ends
    assert not gi.overlaps(ts("10:00"), ts("10:40"))  # the window ends as the gap starts
    assert not gi.overlaps(ts("11:00"), ts("11:30"))


# point-in-time state


def event(tpl, etype, planned, actual, forecast, message, offset):
    kind, when = ("actual", actual) if actual else ("forecast", forecast)
    return {
        "rid": RID, "ssd": pd.Timestamp(TODAY), "tiploc": tpl, "event_type": etype,
        "time_kind": kind, "time": ts(when), "planned_time": ts(planned), "late_reason": None,
        "message_ts": ts(message), "partition": 0, "offset": offset,
    }  # fmt: skip


def service_events(cov_arr_actual="10:28", eus_actual="11:34", cov_message="10:31"):
    """BHM dep 10:00 (2 min late, seen 10:03); Coventry arr 10:30 and dep 10:32; Euston 11:30.

    As in Darwin, a forecast for every calling point arrives when the train is activated (09:30).
    Coventry's arrival is early (10:28) but its message is only sent at `cov_message`, after T for
    the 60 minute horizon (10:30). The HSP rule would count it at T, the Darwin rule must not.
    """
    rows = [
        event("BHAMNWS", "dep", "10:00", None, "10:00", "09:30", 1),
        event("COVNTRY", "arr", "10:30", None, "10:30", "09:30", 2),
        event("COVNTRY", "dep", "10:32", None, "10:32", "09:30", 3),
        event("EUSTON", "arr", "11:30", None, "11:30", "09:30", 4),
        event("BHAMNWS", "dep", "10:00", "10:02", None, "10:03", 5),
        event("COVNTRY", "arr", "10:30", cov_arr_actual, None, cov_message, 6),
        event("COVNTRY", "dep", "10:32", "10:34", None, "10:35", 7),
        event("EUSTON", "arr", "11:30", eus_actual, None, "11:35", 8),
    ]
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def schedule(message, cancel_reason=None, offset=0):
    return {
        "rid": RID, "toc": "VT", "cancel_reason": cancel_reason, "n_stops": 4,
        "n_cancelled_stops": 0, "message_ts": ts(message), "offset": offset,
    }  # fmt: skip


def make_data(events, stamps, schedules=None):
    events = events.sort_values(["message_ts", "offset"], kind="stable").reset_index(drop=True)
    by_rid = {RID: (events, le._ns(events["message_ts"]))}
    versions = schedules if schedules is not None else [schedule("09:00")]
    sched = {RID: (versions, le._ns(pd.Series([v["message_ts"] for v in versions])))}
    return le.ReplayData(by_rid, sched, np.unique(stamps), TIPLOCS)


def test_store_holds_only_messages_at_or_before_t():
    data = make_data(service_events(), minutes("09:00", "11:40"))
    early = data.store_at([RID], ts("10:30"))
    assert set(early.events["offset"]) == {1, 2, 3, 4, 5}  # forecasts and the 10:03 message
    table = early.stop_table(RID)
    assert table.loc[table["location"] == "BHM", "actual_dep"].iloc[0] == ts("10:02")
    assert table.loc[table["location"] == "COVNTRY", "actual_arr"].isna().all()
    # a message stamped exactly T is known at T, the next second is not
    assert set(data.store_at([RID], ts("10:31")).events["offset"]) == {1, 2, 3, 4, 5, 6}
    assert set(data.store_at([RID], ts("10:30:59")).events["offset"]) == {1, 2, 3, 4, 5}
    assert set(data.store_at([RID], None).events["offset"]) == set(range(1, 9))


def test_a_cancellation_sent_after_t_is_unknown_at_t_and_the_age_is_as_of_t():
    versions = [schedule("09:00", offset=0), schedule("10:45", cancel_reason="102", offset=9)]
    data = make_data(service_events(), minutes("09:00", "11:40"), versions)
    at_t = data.store_at([RID], ts("10:30"))
    assert at_t.cancellation(RID) == (False, None)
    assert data.store_at([RID], ts("10:50")).cancellation(RID) == (True, "102")
    assert at_t.newest_message_ts == ts("10:30").to_pydatetime()  # newest captured at or before T
    assert at_t.data_age_minutes(ts("10:30").to_pydatetime()) == 0.0


# replay through the API's own functions


def replay_service(features, bundle_pair):
    bundle, base = bundle_pair
    return le._service_class()(
        store=LiveStore(Path("."), Path("."), TIPLOCS),
        features=features,
        routes=load_routes("config/routes.yaml").routes,
        bands=BANDS,
        rules=load_rules("config/delay_repay.yaml"),
        bundle=bundle,
        baseline=base,
        model_run_id="run-test",
        model_version="7",
        route_model_trained_on={"bhm-eus"},
    )


def universe(eus_actual="11:34", message="11:35"):
    return pd.DataFrame(
        [
            {
                "rid": RID, "route_id": "bhm-eus", "date": TODAY, "toc": "VT",
                "sched_arr": ts("11:30"), "actual_arr": ts(eus_actual),
                "actual_msg_ts": ts(message), "cancelled": False,
                "delay_min": (ts(eus_actual) - ts("11:30")).total_seconds() / 60.0,
            }
        ]
    )  # fmt: skip


def no_gaps():
    return le.GapIndex(
        pd.DataFrame({"gap_start": pd.to_datetime([], utc=True)}).assign(
            gap_end=pd.to_datetime([], utc=True)
        )
    )


def run(svc, data, gaps, uni=None, horizons=(60, 15)):
    return le.replay_singles(svc, data, uni if uni is not None else universe(), gaps, horizons)


def test_features_at_t_use_the_darwin_message_rule_not_the_actual_time_rule(features, bundle_pair):
    svc = replay_service(features, bundle_pair)
    data = make_data(service_events(), minutes("09:00", "11:40"))
    rows, _ = run(svc, data, no_gaps())
    at60 = rows[rows["horizon"] == 60].iloc[0]
    # Coventry's arrival (actual 10:28, before T) arrived in a message after T, so it is unknown
    assert at60["n_stops_passed"] == 1 and at60["last_stop_lateness_min"] == 2.0
    at15 = rows[rows["horizon"] == 15].iloc[0]
    assert at15["n_stops_passed"] == 2  # by 11:15 the BHM and Coventry stops are known
    assert at15["last_stop_lateness_min"] == 2.0  # Coventry departure, two minutes late


def test_no_message_after_t_changes_the_features_or_the_prediction_at_t(features, bundle_pair):
    svc = replay_service(features, bundle_pair)
    stamps = minutes("09:00", "11:40")
    base_rows, _ = run(svc, make_data(service_events(), stamps), no_gaps())
    # change what every message after T=10:30 says: other times, a different final arrival
    poisoned = make_data(
        service_events(cov_arr_actual="10:05", eus_actual="12:59", cov_message="10:31"), stamps
    )
    poisoned_events = poisoned.events[RID][0].copy()
    late = poisoned_events["message_ts"] > ts("10:30")
    poisoned_events.loc[late & (poisoned_events["event_type"] == "dep"), "time"] = ts("10:50")
    poisoned.events[RID] = (poisoned_events, le._ns(poisoned_events["message_ts"]))
    new_rows, _ = run(svc, poisoned, no_gaps(), universe("12:59", "12:59"))
    a = base_rows[base_rows["horizon"] == 60].iloc[0]
    b = new_rows[new_rows["horizon"] == 60].iloc[0]
    for col in ("last_stop_lateness_min", "n_stops_passed", "stops_remaining", "point", "lo",
                "hi", "p15", "p30", "p60"):  # fmt: skip
        assert a[col] == b[col], col
    assert a["delay_min"] != b["delay_min"]  # only the outcome moved


def test_the_replay_matches_what_risk_answers_at_the_same_time(features, bundle_pair):
    svc = replay_service(features, bundle_pair)
    data = make_data(service_events(), minutes("09:00", "11:40"))
    rows, _ = run(svc, data, no_gaps())
    parity = le.parity_with_risk(svc, data, rows, sample=5, seed=0)
    assert parity["checked"] == 2
    assert parity["same_median_to_0.1_min"] == 2 and parity["same_p15_to_4dp"] == 2


def test_service_whose_window_overlaps_a_gap_is_excluded_and_the_rest_are_kept(
    features, bundle_pair
):
    svc = replay_service(features, bundle_pair)
    stamps = minutes("09:00", "11:40", skip=("10:40", "10:47"))
    gaps = le.GapIndex(le.find_gaps(stamps, ts("09:00"), ts("11:40")))
    rows, funnel = run(svc, make_data(service_events(), stamps), gaps)
    status = dict(zip(funnel["horizon"], funnel["status"], strict=True))
    assert status[60] == "gap"  # T=10:30 to 11:30 holds the 10:40 hole
    assert status[15] == "scored"  # T=11:15 to 11:35 does not
    assert list(rows["horizon"]) == [15]


def test_window_runs_to_the_message_that_carried_the_actual_arrival(features, bundle_pair):
    svc = replay_service(features, bundle_pair)
    # the hole is after the scheduled arrival (11:30) but before the actual arrival was reported
    stamps = minutes("09:00", "11:50", skip=("11:31", "11:38"))
    gaps = le.GapIndex(le.find_gaps(stamps, ts("09:00"), ts("11:50")))
    _, funnel = run(svc, make_data(service_events(), stamps), gaps)
    assert set(funnel["status"]) == {"gap"}
    # with the arrival reported before the hole, the same hole is outside both windows
    _, funnel = run(svc, make_data(service_events(), stamps), gaps, universe("11:30", "11:30"))
    assert set(funnel["status"]) == {"scored"}


def test_gap_before_t_is_flagged_without_excluding(features, bundle_pair):
    svc = replay_service(features, bundle_pair)
    stamps = minutes("09:00", "11:40", skip=("10:02", "10:10"))  # well before both T
    gaps = le.GapIndex(le.find_gaps(stamps, ts("09:00"), ts("11:40")))
    _, funnel = run(svc, make_data(service_events(), stamps), gaps)
    flags = dict(zip(funnel["horizon"], funnel["gap_before_T"], strict=True))
    assert set(funnel["status"]) == {"scored"}
    assert flags == {60: True, 15: False}  # 10:30 minus 60 minutes reaches the hole, 11:15 does not


def test_cancelled_and_unknown_services_are_counted_not_predicted(features, bundle_pair):
    svc = replay_service(features, bundle_pair)
    stamps = minutes("09:00", "11:40")
    versions = [schedule("09:00"), schedule("10:20", cancel_reason="102", offset=9)]
    _, funnel = run(svc, make_data(service_events(), stamps, versions), no_gaps())
    assert set(funnel["status"]) == {"cancelled_at_T"}
    # nothing known about the service by T: its schedule and first messages come later
    late = service_events()
    late = late[late["message_ts"] >= ts("11:00")]
    data = make_data(late, stamps, [schedule("11:00")])
    _, funnel = run(svc, data, no_gaps())
    status = dict(zip(funnel["horizon"], funnel["status"], strict=True))
    assert status[60] == "unknown_at_T"  # no message at all about the service by 10:30


# comparing with the test block


def test_verdict_calls_overlapping_intervals_similar():
    live = {"value": 1.0, "ci_low": 0.5, "ci_high": 1.5}
    assert le.verdict(live, {"value": 1.4, "ci_low": 1.2, "ci_high": 1.7}) == "similar"
    assert le.verdict(live, {"value": 3.0, "ci_low": 2.5, "ci_high": 3.5}) == "better"
    assert le.verdict(live, {"value": 0.1, "ci_low": 0.05, "ci_high": 0.2}) == "worse"
    assert le.verdict({"value": 1.0, "ci_low": None, "ci_high": None}, live).startswith(
        "not testable"
    )
    near = {"value": 0.81, "ci_low": 0.8, "ci_high": 0.82}
    far = {"value": 0.7, "ci_low": 0.69, "ci_high": 0.71}
    assert le.verdict(near, far, target=0.8) == "closer to the target"


# scoring


def scored_frame(n=120, seed=0):
    """Replayed rows over six dates with a filtered (non-contiguous) index, as after a mask."""
    rng = np.random.default_rng(seed)
    late = rng.choice([np.nan, 0.0, 3.0, 8.0, 20.0], size=n, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    delay = np.nan_to_num(late) + rng.normal(0, 2, n).round()
    point = np.nan_to_num(late)
    probs = np.tile([0.8, 0.1, 0.05, 0.03, 0.02], (n, 1))
    frame = pd.DataFrame(
        {
            "route_id": "bhm-eus", "dest_hour": 12.0, "hist_median_28d": 2.0, "horizon": 60,
            "last_stop_lateness_min": late, "delay_min": delay,
            "date": [date(2026, 10, 16 + i % 6) for i in range(n)],
            "point": point, "lo": point - 4, "hi": point + 4, "probs": list(probs),
        }
    )  # fmt: skip
    frame.index = np.arange(n) * 3 + 100
    return frame


def test_coverage_by_observed_lateness_groups_rows_by_their_own_lateness():
    from rail_risk.evaluation import HistoricalBaseline, PersistenceIntervalBaseline

    sub = scored_frame()
    train = sub.rename(columns={"delay_min": "label_delay_min", "horizon": "horizon_min"})
    persistence = PersistenceIntervalBaseline(BANDS).fit(train)
    historical = HistoricalBaseline(BANDS).fit(train)
    boot = {"resamples": 50, "seed": 1, "confidence": 0.95}
    cell = le.score_cell(sub, persistence, historical, BANDS, "historical", boot)
    groups = cell["coverage_by_observed_lateness"]
    late = sub["last_stop_lateness_min"]
    assert groups["not_started"]["n"] == late.isna().sum()
    assert groups["0-4"]["n"] == ((late >= 0) & (late < 5) | (late < 0)).sum()
    assert groups["5-14"]["n"] == ((late >= 5) & (late < 15)).sum()
    assert groups["15+"]["n"] == (late >= 15).sum()
    assert cell["not_started_share"] == late.isna().mean()
    assert cell["n_dates"] == 6 and cell["ci_available"]
