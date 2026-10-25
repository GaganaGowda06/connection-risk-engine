from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from rail_risk.baselines import build, gap_bucket, models, pairs, scoring
from rail_risk.baselines.services import route_services, snapshot_rows
from rail_risk.baselines.split import expected_labels, is_complete, network_split
from rail_risk.baselines.state import ENGINES, state_reference
from rail_risk.clean.checks import RouteSpec

UTC = UTC
T0 = datetime(2026, 10, 14, 10, 0, tzinfo=UTC)


def ts(minutes: float) -> pd.Timestamp:
    return pd.Timestamp(T0 + timedelta(minutes=minutes))


def updates_frame() -> pd.DataFrame:
    """Two services. r1 reports an actual at A (3 min late) at 09:20 and one at B (5 late) at
    09:50, and forecasts the destination D at 09:55 (7 late) then at 10:05 (9 late)."""
    rows = []

    def row(rid, tpl, event, plan, actual=None, est=None, msg=0.0):
        rows.append(
            dict(
                rid=rid, tpl=tpl, event=event, pt_arr=ts(plan) if event == "arr" else pd.NaT,
                pt_dep=ts(plan) if event == "dep" else pd.NaT,
                wt_arr=ts(plan) if event == "arr" else pd.NaT,
                wt_dep=ts(plan) if event == "dep" else pd.NaT,
                wt_pass=ts(plan) if event == "pass" else pd.NaT,
                est=ts(est) if est is not None else pd.NaT,
                actual=ts(actual) if actual is not None else pd.NaT,
                message_time=ts(msg),
            )
        )  # fmt: skip

    row("r1", "D", "arr", 60, est=67, msg=-5)  # forecast 7 late at 09:55
    row("r1", "A", "dep", -60, actual=-57, msg=-40)  # 3 late, reported 09:20
    row("r1", "B", "pass", -20, actual=-15, msg=-10)  # 5 late, reported 09:50
    row("r1", "D", "arr", 60, est=69, msg=5)  # forecast 9 late at 10:05
    row("r1", "C", "arr", 30, actual=40, msg=40)  # 10 late, reported later
    row("r2", "D", "arr", 70, msg=-30)
    return pd.DataFrame(rows)


def snaps_frame() -> pd.DataFrame:
    rows = []
    for rid, h in (("r1", 60), ("r1", 30), ("r2", 60)):
        pt = ts(60) if rid == "r1" else ts(70)
        rows.append(
            dict(rid=rid, route="rt", horizon_min=h, snap_time=pt - timedelta(minutes=h),
                 dest_tpl="D", pt_arr=pt)
        )  # fmt: skip
    return pd.DataFrame(rows)


def test_state_uses_only_messages_at_or_before_snapshot_time():
    # r1 arrives 11:00. At 60 min (T = 10:00) the 10:05 forecast and the 10:40 actual are later.
    out = state_reference(updates_frame(), snaps_frame())
    r = out[(out.rid == "r1") & (out.horizon_min == 60)].iloc[0]
    assert (r.last_actual_tpl, r.last_actual_late_s, r.fc_late_s) == ("B", 300, 7 * 60)
    r = out[(out.rid == "r1") & (out.horizon_min == 30)].iloc[0]  # T = 10:30
    assert (r.last_actual_tpl, r.last_actual_late_s, r.fc_late_s) == ("B", 300, 9 * 60)
    r = out[out.rid == "r2"].iloc[0]  # nothing reported yet
    assert pd.isna(r.last_actual_late_s) and pd.isna(r.fc_late_s)


def test_deleting_messages_after_t_changes_nothing():
    u, s = updates_frame(), snaps_frame()
    full = state_reference(u, s)
    for snap in s.itertuples(index=False):
        cut = u[u["message_time"] <= snap.snap_time]
        one = state_reference(cut, s[(s.rid == snap.rid) & (s.horizon_min == snap.horizon_min)])
        row = full[(full.rid == snap.rid) & (full.horizon_min == snap.horizon_min)]
        pd.testing.assert_frame_equal(one.reset_index(drop=True), row.reset_index(drop=True))


@pytest.mark.parametrize("engine", ["duckdb", "polars", "spark"])
def test_engines_match_the_reference(engine):
    u, s = updates_frame(), snaps_frame()
    got = ENGINES[engine](pa.Table.from_pandas(u, preserve_index=False), s)
    pd.testing.assert_frame_equal(got, state_reference(u, s), check_dtype=False)


def points_frame() -> pd.DataFrame:
    rows = []
    for rid, seq, tpl, arr, dep in (
        ("a", 0, "O1", None, 0), ("a", 1, "MID", 10, 11), ("a", 2, "D1", 20, None),
        ("b", 0, "D1", None, 0), ("b", 1, "O1", 5, None),
    ):  # fmt: skip
        rows.append(
            dict(rid=rid, seq=seq, tpl=tpl, pt_arr=ts(arr) if arr is not None else pd.NaT,
                 pt_dep=ts(dep) if dep is not None else pd.NaT, message_time=ts(-1000))
        )  # fmt: skip
    return pd.DataFrame(rows)


def test_services_need_origin_before_destination_and_use_first_schedule():
    sched = pd.DataFrame(
        {"rid": ["a", "b"], "toc": ["XX", "XX"], "train_cat": ["XX", "XX"],
         "is_passenger": [True, True], "message_time": [ts(-1000)] * 2}
    )  # fmt: skip
    spec = RouteSpec("r", frozenset({"O1"}), frozenset({"D1"}))
    out = route_services(sched, points_frame(), [spec], date(2026, 10, 14))
    assert list(out["rid"]) == ["a"]  # b runs D1 then O1, the wrong way round
    snaps = snapshot_rows(out)
    assert sorted(snaps["horizon_min"]) == [5, 15, 30, 60, 90, 120]
    row = snaps[snaps.horizon_min == 30].iloc[0]
    assert row["snap_time"] == ts(20) - timedelta(minutes=30) and row["schedule_known"]


def test_replacement_services_are_not_trains():
    sched = pd.DataFrame(
        {"rid": ["a"], "toc": ["XX"], "train_cat": ["BR"], "is_passenger": [True],
         "message_time": [ts(-1000)]}
    )  # fmt: skip
    spec = RouteSpec("r", frozenset({"O1"}), frozenset({"D1"}))
    assert route_services(sched, points_frame(), [spec], date(2026, 10, 14)).empty


def snap_table(n_days=8, per_day=60, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for i in range(per_day):
            late = rng.gamma(1.5, 4.0)
            for h in (15, 60):
                known = max(0.0, late * rng.uniform(0.5, 1.1) + rng.normal(0, 1))
                rows.append(
                    dict(ssd=date(2026, 9, 1) + timedelta(days=d), rid=f"{d}-{i}",
                         route="rt" if i % 4 else "ho", horizon_min=h, arr_hour_local=8 + i % 6,
                         last_actual_late_s=known * 60, fc_late_s=known * 60,
                         target_late_s=late * 60, has_target=True, schedule_known=True)
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_baselines_fit_predict_score_end_to_end():
    train, test = snap_table(seed=1), snap_table(n_days=4, seed=2)
    fitted = models.fit(train)
    preds = models.predict(fitted, test)
    assert set(preds) == set(models.METHODS)
    for p in preds.values():
        assert (p["lo"] <= p["hi"]).all() and p["p_late"].between(0, 1).all()
    stats = {m: scoring.row_stats(test, p) for m, p in preds.items()}
    out = scoring.score_all(stats, {"ho"}, ["rt", "ho", "none"], 200, 1, 0.95)
    one = out[(out.method == "persistence") & (out.group == "all") & (out.horizon_min == 15)].iloc[
        0
    ]
    assert one["mae_lo"] <= one["mae"] <= one["mae_hi"] and one["n_dates"] == 4
    # persistence knows the lateness here, the historical baseline does not
    hist = out[(out.method == "historical") & (out.group == "all") & (out.horizon_min == 15)].iloc[
        0
    ]
    assert one["mae"] < hist["mae"]
    # a route with no rows is reported as empty, not an error
    assert int(out[(out.group == "none")]["n"].max()) == 0
    # coverage of an 80% range fitted on the same process is near 80%
    assert 0.7 < hist["coverage"] < 0.9


def test_historical_cell_falls_back_when_thin():
    train = snap_table(n_days=1, per_day=8)
    fitted = models.fit(train)
    test = train.assign(route="unseen")
    p = models.predict(fitted, test)["historical"]
    assert np.allclose(p["point"], fitted.hist_all[1])  # network fallback, never an error


def test_fit_ignores_rows_without_target_or_known_schedule():
    train = snap_table()
    bad = train.copy()
    bad["has_target"] = False
    with_bad = pd.concat([train, bad])
    assert models.fit(with_bad).fit_rows == models.fit(train).fit_rows


def test_gap_bucket_needs_a_parsed_mct_and_labels_misses(tmp_path):
    with pytest.raises(gap_bucket.MctMissing):
        gap_bucket.load_mct_table(tmp_path / "none.csv")
    t = pd.Timestamp("2026-10-14 10:00", tz="UTC")
    m = pd.Timedelta(minutes=1)
    pairs = pd.DataFrame(
        {
            "hub": ["H"] * 3,
            "mct_min": [10.0] * 3,
            "plan_arr_in": [t] * 3,
            "plan_dep_out": [t + 12 * m, t + 12 * m, t + 12 * m],
            "act_arr_in": [t, t + 6 * m, t],
            "act_dep_out": [t + 12 * m, t + 12 * m, pd.NaT],
        }
    )
    out = gap_bucket.label_pairs(pairs)  # slack 2 minutes
    assert list(out["miss"]) == [0.0, 1.0, 1.0]  # late inbound misses; a cancelled outbound too
    model = gap_bucket.fit_buckets(out)
    assert 0 < gap_bucket.predict_buckets(model, out)[0] < 1


def test_expected_labels_and_complete_days():
    assert len(expected_labels(date(2026, 3, 29))) == 23 and 1 not in expected_labels(
        date(2026, 3, 29)
    )
    assert len(expected_labels(date(2026, 10, 25))) == 24
    d = date(2026, 5, 5)
    files = {(d + timedelta(days=k), h) for k in (-1, 0) for h in range(24)}
    files |= {(d + timedelta(days=1), h) for h in range(5)}
    assert is_complete(d, files)
    assert not is_complete(d, files - {(d, 13)})


def test_shipped_network_split_is_ordered_gapped_and_train_pool_only():
    split, _ = network_split()
    blocks = split.blocks()
    assert (len(blocks["test"]), len(blocks["validation"])) == (7, 14)
    assert max(blocks["train"]) < min(blocks["validation"]) - timedelta(days=1)
    assert max(blocks["validation"]) < min(blocks["test"]) - timedelta(days=1)
    every = [d for b in blocks.values() for d in b]
    assert not any(date(2026, 2, 1) <= d <= date(2026, 2, 28) for d in every)  # winter_test
    assert not any(date(2026, 10, 17) <= d <= date(2026, 10, 23) for d in every)  # oot_week


def test_split_yaml_dates_match_the_rule():
    import yaml

    split, _ = network_split()
    stated = yaml.safe_load(open("config/split.yaml"))["network"]
    for name, days in split.blocks().items():
        assert (stated[name]["first"], stated[name]["last"], stated[name]["days"]) == (
            days[0],
            days[-1],
            len(days),
        )


def test_committed_leakage_report_shows_zero_mismatches_on_real_rows():
    import json

    rep = json.load(open("reports/stage2_leakage.json"))
    assert rep["snapshots_checked"] >= 500 and rep["snapshots_with_an_actual_by_T"] > 100
    assert rep["mismatches"] == 0 and not any(rep["mismatches_by_column"].values())


def test_committed_run_used_only_train_pool_days():
    import json

    meta = json.load(open("reports/baselines/stage2_baselines.json"))
    assert meta["oot_week_or_winter_rows_in_fit"] == 0
    failure = json.load(open("reports/stage2_failure_tests.json"))
    assert failure["oot_week_or_winter_test_dates_in_snapshots"] == 0
    assert failure["fit_dates_outside_train_pool"] == 0


def test_shrunk_persistence_weight_is_fitted_on_train_and_bounded():
    train = snap_table(seed=1)
    fitted = models.fit(train)
    assert set(fitted.shrink_w) == {15, 60}
    assert all(0.0 <= w <= 1.0 for w in fitted.shrink_w.values())
    # pure persistence is exact when the state equals the outcome, so no shrinkage is chosen
    exact = train.assign(last_actual_late_s=train["target_late_s"])
    assert models.fit(exact).shrink_w == {15: 1.0, 60: 1.0}
    # a state with no information gets pulled all the way to the history
    blind = train.assign(last_actual_late_s=0.0)
    assert min(models.fit(blind).shrink_w.values()) < 0.5


def test_pair_state_flags_use_only_messages_at_or_before_forecast_time():
    t = pd.Timestamp("2026-10-14 10:00", tz="UTC")
    m = pd.Timedelta(minutes=1)
    p = pd.DataFrame(
        {
            "plan_arr_in": [t, t],
            "in_arr_msg_time": [t - 20 * m, t - 5 * m],  # arrived 20 and 5 minutes early
            "in_seen": [t - 90 * m, t - 10 * m],
            "out_seen": [t - 90 * m, t - 90 * m],
        }
    )
    k = pairs.known_at(p, 15)
    assert list(k["inbound_arrived_by_t"]) == [True, False]
    assert list(k["schedule_known"]) == [True, False]  # second inbound schedule seen after T


def test_target_is_the_arrival_nearest_the_plan_not_the_latest():
    t0 = pd.Timestamp("2026-09-25 13:25", tz="UTC")
    out = pd.DataFrame(
        {
            "rid": ["a", "b", "c"],
            "route": "r",
            "horizon_min": 15,
            "dest_tpl": "X",
            "pt_arr": [t0] * 3,
        }
    )
    day = pd.Timedelta(days=1)
    targets = pd.DataFrame(
        {
            "rid": ["a", "a", "b", "c"],
            "dest_tpl": "X",
            "actual_arr": [t0 + pd.Timedelta(minutes=1), t0 + day, t0 + day, pd.NaT],
        }
    )
    got = build.pick_targets(out, targets).set_index("rid")["actual_arr"]
    assert got["a"] == t0 + pd.Timedelta(minutes=1)  # the day-later duplicate is ignored
    assert pd.isna(got["b"]) and pd.isna(got["c"])  # only a day-late row, or none: no target
