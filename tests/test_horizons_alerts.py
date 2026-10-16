"""Horizon buckets, route-aware Mondrian offsets, alert thresholds and the backtest helpers."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rail_risk.alerts import (
    NEVER,
    AlertConfig,
    alert_report,
    choose_threshold,
    load_alert_config,
    policy_stats,
    threshold_grid,
    trivial_policies,
)
from rail_risk.backtest import MissBaseline, gap_bucket, inbound_state
from rail_risk.connection import (
    FixedDelays,
    Train,
    simulate,
)
from rail_risk.connection_report import (
    END,
    START,
    alert_inputs,
    connections_section,
    write_connections_section,
)
from rail_risk.horizons import HORIZON_GRID, bucket_labels, horizon_buckets
from rail_risk.models.quantile import fit_quantile
from test_models import CFG


def test_horizon_buckets_cover_the_grid_and_any_real_minutes():
    assert HORIZON_GRID == (5, 15, 30, 60, 90, 120)
    assert horizon_buckets(list(HORIZON_GRID)).tolist() == [0, 0, 1, 1, 2, 2]
    # between grid points, and below and above the trained range
    assert horizon_buckets([-12, 0, 19, 20, 47, 74, 75, 400]).tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert bucket_labels() == ["under 20", "20 to 74", "75 or more"]


def route_frame(n: int, seed: int) -> pd.DataFrame:
    """Two routes whose noise differs, three horizons, a model blind to route."""
    rng = np.random.default_rng(seed)
    route = rng.choice(["quiet", "noisy"], n)
    horizon = rng.choice([15, 60, 120], n)
    scale = np.where(route == "noisy", 6.0, 1.0)
    y = rng.normal(0, 1, n) * scale
    return pd.DataFrame(
        {
            "route_id": route,
            "horizon_min": horizon,
            "label_delay_min": y,
            "service_date": [date(2026, 1, 1) + timedelta(days=int(i % 10)) for i in range(n)],
        }
    )


def test_offsets_are_per_route_and_cover_each_route_and_bucket():
    df = route_frame(60_000, seed=11)
    train, calib, test = df.iloc[:20_000], df.iloc[20_000:40_000], df.iloc[40_000:]
    # blind to route on purpose: only the per-route offsets can correct each route's coverage
    fit, _ = fit_quantile(train, calib, calib, CFG, columns=["horizon_min"])
    _, lo, hi = fit.predict(test)
    y = test["label_delay_min"].to_numpy()
    buckets = horizon_buckets(test["horizon_min"].to_numpy())
    for route in ("quiet", "noisy"):
        for b in (0, 1, 2):
            m = (test["route_id"] == route).to_numpy() & (buckets == b)
            assert m.sum() > 1500
            assert abs(float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))) - 0.8) < 0.035, (route, b)
    # the two routes got clearly different offsets
    assert abs(fit.offsets[("noisy", 1, 0)] - fit.offsets[("quiet", 1, 0)]) > 1.0


def test_unseen_route_and_off_grid_minutes_use_pooled_offsets_and_the_right_bucket():
    df = route_frame(20_000, seed=2)
    fit, _ = fit_quantile(
        df.iloc[:10_000], df.iloc[10_000:], df.iloc[10_000:], CFG, ["horizon_min"]
    )
    new = pd.DataFrame(
        {"route_id": ["elsewhere", "quiet"], "horizon_min": [47, 47], "label_delay_min": 0.0}
    )
    routes, buckets, groups = fit._cells(new)
    assert routes.tolist() == ["*", "quiet"] and buckets.tolist() == [1, 1]
    point, lo, hi = fit.predict(new)
    assert (lo <= point).all() and (point <= hi).all()
    _, q = fit.distribution(new)
    assert q.shape == (2, len(fit.quantiles)) and (np.diff(q, axis=1) >= 0).all()


# alerts


def calibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 4.0, n)
    return p, (rng.random(n) < p).astype(float)


def test_policy_stats_by_hand():
    p = np.array([0.9, 0.6, 0.3, 0.1])
    y = np.array([1, 0, 1, 0])
    s = policy_stats(p, y, 0.5, false_alert=1.0, missed_alert=5.0)
    assert (s["alerts"], s["true_alerts"], s["false_alerts"], s["missed_alerts"]) == (2, 1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["alert_rate"] == 0.5
    assert s["cost_per_pair"] == pytest.approx((1 * 1 + 5 * 1) / 4)
    assert policy_stats(p, y, NEVER, 1.0, 5.0)["precision"] is None  # no alerts, no precision


def test_trivial_policies_cost():
    p, y = calibrated(2000, 1)
    t = trivial_policies(p, y, 1.0, 5.0)
    assert t["always_alert"]["recall"] == 1.0 and t["always_alert"]["alert_rate"] == 1.0
    assert t["never_alert"]["alerts"] == 0
    assert t["never_alert"]["cost_per_pair"] == pytest.approx(5.0 * y.mean())
    assert t["always_alert"]["cost_per_pair"] == pytest.approx(1.0 * (1 - y.mean()))


def test_threshold_is_the_cost_minimiser_and_matches_theory_when_calibrated():
    p, y = calibrated(200_000, 3)
    tau = choose_threshold(p, y, 1.0, 5.0, 0.01)
    assert abs(tau - 1.0 / 6.0) < 0.03  # c_fa / (c_fa + c_miss) for a calibrated forecast
    costs = {t: policy_stats(p, y, t, 1.0, 5.0)["cost_per_pair"] for t in threshold_grid(0.01)}
    assert costs[tau] == pytest.approx(min(costs.values()))


def test_threshold_falls_as_missed_alerts_cost_more_and_ties_go_to_fewer_alerts():
    p, y = calibrated(50_000, 4)
    taus = [choose_threshold(p, y, 1.0, c, 0.01) for c in (3.0, 5.0, 10.0)]
    assert taus[0] >= taus[1] >= taus[2]
    # nothing ever happens: never alerting is cheapest and the tie-break picks it
    assert choose_threshold(np.array([0.3, 0.5]), np.zeros(2), 1.0, 5.0) == NEVER


def test_test_rows_never_move_the_threshold():
    pv, yv = calibrated(5000, 5)
    pt, yt = calibrated(3000, 6)
    cfg = AlertConfig(1.0, 5.0, (3.0, 10.0), 0.01)
    a = alert_report("x", (pv, yv), (pt, yt), cfg)
    b = alert_report("x", (pv, yv), (pt, 1.0 - yt), cfg)  # test outcomes flipped
    assert (
        a["tau"] == b["tau"]
        and a["sensitivity_to_missed_alert_cost"] == (b["sensitivity_to_missed_alert_cost"])
    )
    assert a["test"]["chosen"] != b["test"]["chosen"]  # but the scored result moves
    assert set(a["sensitivity_to_missed_alert_cost"]) == {"5", "3", "10"}


def test_alert_config_is_read_and_labelled_as_assumptions():
    cfg = load_alert_config("config/alerts.yaml")
    assert (cfg.false_alert, cfg.missed_alert) == (1.0, 5.0)
    text = open("config/alerts.yaml", encoding="utf-8").read().lower()
    assert text.count("assumption") >= 3 and "product input" in text


# backtest helpers


def test_gap_buckets_and_baseline_use_training_pairs_only():
    assert gap_bucket([10, 14, 15, 24, 25, 90], 10.0).tolist() == [0, 0, 1, 2, 3, 3]
    train = pd.DataFrame({"gap": [10, 12, 16, 30], "missed": [1.0, 0.0, 0.0, np.nan]})
    base = MissBaseline(train, 10.0)
    assert base.overall == pytest.approx(1 / 3)
    assert base.predict([11, 17, 80]).tolist() == pytest.approx(
        [0.5, 0.0, 1 / 3]
    )  # unseen: overall
    with pytest.raises(ValueError):
        MissBaseline(train.assign(missed=np.nan), 10.0)


def pair(**kw):
    base = dict(planned_dep=600.0, inbound_sched_arr=585.0, inbound_actual_arr=587.0)
    return SimpleNamespace(**{**base, **kw})


def test_inbound_state_uses_the_exact_minutes_at_forecast_time():
    # forecast 60 minutes before a 10:00 departure is 09:00; due 09:45, so 45 minutes to go
    assert inbound_state(pair(), 60) == ("approaching", 45)
    # 15 minutes before is 09:45: due now, not yet arrived (arrives 09:47), so due or late
    assert inbound_state(pair(), 15) == ("due_or_late", 0)
    assert inbound_state(pair(inbound_actual_arr=584.0), 15) == ("arrived", None)
    # forecast at 09:45 sharp (departure 09:55, k=10) is zero minutes; at 09:50 it is minus five
    assert inbound_state(pair(inbound_actual_arr=np.nan, planned_dep=595.0), 10) == (
        "due_or_late",
        0,
    )
    assert inbound_state(pair(inbound_actual_arr=np.nan), 10) == ("due_or_late", -5)
    # 5 minutes to go is the shortest trained horizon; 4 minutes is below it
    assert inbound_state(pair(inbound_actual_arr=np.nan), 20) == ("approaching", 5)
    assert inbound_state(pair(inbound_actual_arr=np.nan), 19) == ("due_or_late", 4)


def test_inbound_floor_conditions_the_draw_on_not_having_arrived():
    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    q = np.array([-1.0, 0.0, 2.0, 5.0, 12.0])
    planned, out = Train(600.0, 700.0), FixedDelays(0.0, 0.0)
    free = simulate(levels, q, 590.0, planned, [], 10.0, out, draws=4000, seed=1)
    late = simulate(levels, q, 590.0, planned, [], 10.0, out, draws=4000, seed=1, inbound_floor=3.0)
    assert late.p_miss > free.p_miss  # conditioning on a late train raises the miss chance
    # with a fixed outbound at 10:00 and a 10 minute connection, a miss needs a delay over zero;
    # with a floor of 3 every draw is at least 3, so every draw is a miss
    certain = simulate(
        levels, q, 590.0, planned, [], 7.0, out, draws=2000, seed=1, inbound_floor=3.5
    )
    assert certain.p_miss == 1.0 or certain.p_miss > 0.99 and certain.p_miss <= 1.0
    # a floor beyond the whole distribution is held at the top level, never an error
    simulate(levels, q, 590.0, planned, [], 10.0, out, draws=100, seed=1, inbound_floor=500.0)


def predictions(n_per_block: int = 40) -> pd.DataFrame:
    rows = []
    for block in ("validation", "test"):
        for i in range(n_per_block):
            for k in (60, 15):
                rows.append(
                    {
                        "block": block, "date": date(2026, 10, 1 + i % 5), "k": k,
                        "p_miss": 0.1 + 0.01 * i, "p_final_ge15": 0.2, "missed": float(i % 4 == 0),
                        "stranded": i == 3, "final_delay": 20.0 if i % 2 else 3.0,
                        "state": "approaching", "minutes_to_arrival": 40, "lo": 0.0, "hi": 1.0,
                        "realised_arrival": 1.0, "baseline": 0.1, "bucket": 0,
                        "inbound_rid": "a", "outbound_rid": "b", "gap": 12.0, "pair": i,
                        "p_stranded": 0.0,
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_alert_inputs_define_the_outcomes_and_split_blocks():
    pred = predictions()
    (pv, yv), (pt, yt), per_k = alert_inputs(pred, "p_miss", "missed")
    assert len(pv) == len(yv) == 80 and len(pt) == len(yt) == 80
    assert set(per_k) == {"T-60 before outbound", "T-15 before outbound"}
    (_, y15), _, _ = alert_inputs(pred, "p_final_ge15", "delay15")
    # stranded counts as late (1); otherwise at least 15 minutes late is 1
    sub = pred[pred["block"] == "validation"]
    expected = np.where(sub["stranded"], 1.0, (sub["final_delay"] >= 15).astype(float))
    assert y15.tolist() == expected.tolist()


def test_connections_section_is_written_in_place_and_has_no_em_dash(tmp_path):
    result = {
        "ast_bhm_days": {"train": 100, "validation": 14, "test": 7},
        "pairs": {"train": 900, "validation": 100, "test": 50},
        "pairs_with_known_outcome": {"train": 800, "validation": 90, "test": 45},
        "min_connection_time": 10.0, "min_connection_time_status": "assumed",
        "window_minutes": 20.0, "min_trained_minutes": 5,
        "baseline": {"overall_miss_rate": 0.06, "pairs_by_gap_bucket": {"0": 400, "1": 400}},
        "scores": {},
    }  # fmt: skip
    path = tmp_path / "model_results.md"
    path.write_text("# Model results\n\nbody\n", encoding="utf-8")
    write_connections_section(result, path)
    first = path.read_text(encoding="utf-8")
    write_connections_section(result, path)
    assert path.read_text(encoding="utf-8") == first  # a rerun replaces, it does not append
    assert first.startswith("# Model results") and first.count(START) == 1
    assert first.count(END) == 1
    assert "—" not in connections_section(result)


def test_a_tie_between_ready_time_and_departure_is_a_catch_not_half_a_miss():
    # delays are whole minutes in the data. The inbound is about three minutes late (spread a
    # little around it by interpolation) and the outbound leaves exactly three minutes late, so
    # the passenger is ready at the minute the train goes: a catch.
    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    q = np.array([2.6, 2.8, 3.0, 3.2, 3.4])
    res = simulate(
        levels, q, 590.0, Train(600.0, 700.0), [], 10.0, FixedDelays(3.0, 0.0), draws=4000, seed=3
    )
    assert res.p_miss == 0.0
    assert (res.delays == np.round(res.delays)).all()
