import numpy as np
import pandas as pd
import pytest

from rail_risk.connection import (
    FixedDelays,
    InfeasibleConnection,
    OutboundDelays,
    TailModel,
    Train,
    band_probabilities,
    choose_outbound,
    first_feasible,
    hhmm_to_minutes,
    inbound_delay_draws,
    journey_repay,
    load_stations,
    minutes_to_hhmm,
    simulate,
)
from rail_risk.repay import load_rules

LEVELS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
EDGES = (0, 15, 30, 60, 120)
PLANNED = Train(sched_dep=600.0, sched_arr=700.0, toc="VT")  # 10:00 to 11:40
ALTS = [Train(620.0, 720.0, "VT"), Train(640.0, 740.0, "VT"), Train(660.0, 760.0, "VT")]


def flat(delay: float) -> np.ndarray:
    return np.full(len(LEVELS), float(delay))


def spread(median: float, width: float) -> np.ndarray:
    return median + width * (LEVELS - 0.5) * 4


def run(quantiles, mct=10.0, alts=ALTS, outbound=None, inbound_arr=590.0, **kw):
    return simulate(
        LEVELS, quantiles, inbound_arr, PLANNED, alts, mct, outbound or FixedDelays(),
        EDGES, **kw,
    )  # fmt: skip


def test_zero_variance_on_time_connection_is_caught():
    # inbound due 09:50, arrives on time at 09:50, 10 min to connect, train leaves at 10:00
    r = run(flat(0.0))
    assert r.p_miss == 0.0 and r.p_stranded == 0.0 and not r.stranded
    assert r.delay_quantiles == {0.1: 0.0, 0.5: 0.0, 0.9: 0.0}
    assert r.band_probs[0] == 1.0


def test_zero_variance_miss_falls_back_by_exactly_the_headway():
    # arrives 09:56, 10 min to connect: ready 10:06 after the 10:00 departure, catches 10:20
    r = run(flat(6.0))
    assert r.p_miss == 1.0 and r.p_stranded == 0.0
    assert r.delay_quantiles[0.5] == 20.0  # the headway between 10:00 and 10:20
    assert r.arrival_range() == (720.0, 720.0)
    # a worse delay skips the 10:20 train too and lands on the 10:40
    r2 = run(flat(25.0))
    assert r2.delay_quantiles[0.5] == 40.0


def test_connection_exactly_at_the_minimum_time_is_caught():
    assert run(flat(0.0), mct=10.0).p_miss == 0.0  # ready at 10:00, departs 10:00
    assert run(flat(0.0), mct=10.5).p_miss == 1.0


def test_p_miss_is_monotone_in_connection_time_and_in_inbound_lateness():
    outbound = OutboundDelays({10: np.column_stack([np.arange(40) * 0.25, np.arange(40) * 0.5])})
    q = spread(2.0, 6.0)
    by_mct = [run(q, mct=m, outbound=outbound).p_miss for m in (2, 5, 8, 10, 15, 25)]
    assert by_mct == sorted(by_mct) and by_mct[0] < by_mct[-1]
    by_late = [run(q + s, outbound=outbound).p_miss for s in (-5, 0, 3, 6, 12, 30)]
    assert by_late == sorted(by_late) and by_late[0] < by_late[-1]


def test_stranded_draws_count_in_the_top_tier_and_have_no_arrival():
    rules = load_rules("config/delay_repay.yaml")
    r = run(flat(90.0), alts=[])
    assert r.stranded and r.p_stranded == 1.0 and r.p_miss == 1.0
    assert r.delay_quantiles is None and r.arrival_range() is None  # no arrival time to give
    assert r.band_probs.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]  # priced in the top tier
    pay = journey_repay(r, EDGES, "VT", "single", 50.0, rules)
    assert pay.expected_payout == pytest.approx(50.0)  # 100% of a single in the 120+ tier
    # alternatives exist but all leave before the inbound can connect
    assert run(flat(200.0)).stranded


def test_partly_stranded_is_unconditional_not_conditional_on_completing():
    rules = load_rules("config/delay_repay.yaml")
    q = spread(30.0, 40.0)  # the late end cannot reach the last alternative at 10:20
    r = run(q, alts=ALTS[:1])
    assert 0.0 < r.p_stranded < 1.0 and not r.stranded
    assert r.band_probs.sum() == pytest.approx(1.0)
    assert len(r.delays) == round((1 - r.p_stranded) * r.n_draws)
    assert r.band_probs[-1] >= r.p_stranded - 1e-12  # stranded draws sit in the top band
    unconditional = journey_repay(r, EDGES, "VT", "single", 50.0, rules).expected_payout
    completed_only = band_probabilities(r.delays, EDGES)
    cond = float(completed_only @ np.array([0.0, 12.5, 25.0, 50.0, 50.0]))
    assert unconditional > cond  # stranded draws add the top tier, so conditional understated it


def test_band_probabilities_sum_to_one_and_use_the_band_edges():
    p = band_probabilities(
        np.array([-3.0, 0.0, 14.9, 15.0, 29.0, 30.0, 60.0, 119.0, 120.0, 400.0]), EDGES
    )
    assert p.sum() == pytest.approx(1.0)
    assert p.tolist() == pytest.approx([0.3, 0.2, 0.1, 0.2, 0.2])
    r = run(
        spread(8.0, 14.0), outbound=OutboundDelays({10: np.array([[0.0, 0.0], [3.0, 5.0]] * 20)})
    )
    assert r.band_probs.sum() == pytest.approx(1.0)


def test_same_seed_same_result_and_different_seed_differs():
    outbound = OutboundDelays({10: np.column_stack([np.arange(60) * 0.3, np.arange(60) * 0.6])})
    q = spread(4.0, 10.0)
    a, b = run(q, outbound=outbound, seed=5), run(q, outbound=outbound, seed=5)
    assert a.p_miss == b.p_miss and np.array_equal(a.delays, b.delays)
    c = run(q, outbound=outbound, seed=6)
    assert not np.array_equal(a.delays, c.delays)


def test_inbound_draws_are_monotone_and_follow_the_quantiles():
    u = np.linspace(0.001, 0.999, 999)
    x = inbound_delay_draws(LEVELS, spread(5.0, 12.0), u)
    assert (np.diff(x) >= -1e-12).all()
    median = inbound_delay_draws(LEVELS, spread(5.0, 12.0), np.array([0.5]))[0]
    assert median == pytest.approx(5.0)
    # unsorted quantiles are made monotone rather than producing a decreasing inverse CDF
    bad = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 8.0, 9.0])
    y = inbound_delay_draws(LEVELS, bad, u)
    assert (np.diff(y) >= -1e-12).all()


def test_outbound_pairs_stay_together_and_thin_hours_fall_back():
    rich = np.column_stack([np.full(40, 4.0), np.full(40, 9.0)])
    thin = np.array([[100.0, 200.0]] * 3)
    model = OutboundDelays({9: rich, 10: thin})
    rng = np.random.default_rng(0)
    d, a = model.sample(9, rng, 50)
    assert (d == 4.0).all() and (a == 9.0).all()
    d, a = model.sample(10, rng, 200)  # 3 rows is too few: pooled history is used
    assert set(zip(d.tolist(), a.tolist(), strict=True)) <= {(4.0, 9.0), (100.0, 200.0)}
    frame = pd.DataFrame({"hour": [9] * 40, "dep_delay": [1.0] * 40, "arr_delay": [2.0] * 40})
    assert OutboundDelays.fit(frame).sample(9, rng, 3)[1].tolist() == [2.0, 2.0, 2.0]
    with pytest.raises(ValueError):
        OutboundDelays({})


def test_expected_payout_from_the_journey_distribution():
    rules = load_rules("config/delay_repay.yaml")
    r = run(flat(6.0))  # certain 20 minute delay: the 15-29 tier, 25% of a 50 pound single
    pay = journey_repay(r, EDGES, "VT", "single", 50.0, rules)
    assert pay.expected_payout == pytest.approx(12.5)


def test_station_minimum_connection_time_is_labelled():
    bhm = load_stations("config/stations.yaml")["BHM"]
    assert bhm.minutes == 10 and bhm.status == "assumed" and bhm.comment


def test_clock_helpers():
    assert hhmm_to_minutes("11:42") == 702.0
    assert minutes_to_hhmm(702.4) == "11:42"
    assert minutes_to_hhmm(24 * 60 + 5) == "00:05"
    with pytest.raises(ValueError):
        hhmm_to_minutes("25:00")


def test_first_feasible_outbound_is_chosen_when_none_is_given():
    trains = [Train(587.0, 666.0), Train(606.0, 737.0), Train(617.0, 698.0), Train(636.0, 770.0)]
    assert first_feasible(580.0, trains, 10.0).sched_dep == 606.0  # 09:47 leaves only 7 min after
    chosen = choose_outbound(580.0, trains, 10.0)
    assert chosen.sched_dep == 606.0
    assert first_feasible(580.0, trains, 7.0).sched_dep == 587.0  # a shorter change opens 09:47
    assert first_feasible(700.0, trains, 10.0) is None
    with pytest.raises(InfeasibleConnection, match="no outbound"):
        choose_outbound(700.0, trains, 10.0)


def test_infeasible_connection_is_refused_naming_the_first_feasible_train():
    trains = [Train(587.0, 666.0), Train(606.0, 737.0)]  # 09:47 and 10:06
    with pytest.raises(InfeasibleConnection) as err:
        choose_outbound(580.0, trains, 10.0, planned_dep=587.0)
    assert "infeasible on paper" in str(err.value) and "10:06" in str(err.value)
    assert err.value.first_feasible.sched_dep == 606.0
    assert choose_outbound(580.0, trains, 10.0, planned_dep=606.0).sched_dep == 606.0
    assert choose_outbound(580.0, trains, 7.0, planned_dep=587.0).sched_dep == 587.0
    with pytest.raises(InfeasibleConnection, match="no planned outbound"):
        choose_outbound(580.0, trains, 10.0, planned_dep=600.0)


def test_sensitivity_across_connection_times_is_monotone():
    outbound = OutboundDelays({10: np.column_stack([np.arange(40) * 0.2, np.arange(40) * 0.4])})
    q = spread(1.0, 8.0)
    # inbound due 09:35, outbound 10:00: gap 25 min, so all four times are feasible
    p = [run(q, mct=m, outbound=outbound, inbound_arr=575.0).p_miss for m in (5, 7, 10, 15)]
    assert p == sorted(p) and p[0] < p[-1]
    pay = []
    rules = load_rules("config/delay_repay.yaml")
    for m in (5, 7, 10, 15):
        r = run(q, mct=m, outbound=outbound, inbound_arr=575.0)
        pay.append(journey_repay(r, EDGES, "VT", "single", 50.0, rules).expected_payout)
    assert pay == sorted(pay)


def test_tail_sampler_uses_only_the_rows_it_is_given_and_extends_beyond_q95():
    # training rows: outcomes 0 to 30 minutes above the top quantile in group 3, none in group 0
    outcome = np.concatenate([np.full(50, 10.0) + np.arange(50) * 0.6, np.full(50, 5.0)])
    top = np.concatenate([np.full(50, 10.0), np.full(50, 10.0)])
    groups = np.array([3] * 50 + [0] * 50)
    tail = TailModel.fit(outcome, top, groups)
    assert tail.for_group(3).max() == pytest.approx(29.4)
    assert len(tail.for_group(0)) == len(tail.pooled)  # no excesses: falls back to pooled
    u = np.array([0.5, 0.96, 0.99, 0.9999])
    x = inbound_delay_draws(LEVELS, spread(5.0, 12.0), u, tail.for_group(3))
    top_q = spread(5.0, 12.0)[-1]
    assert x[0] == pytest.approx(5.0)
    assert (np.diff(x) >= 0).all() and x[-1] > top_q + 25  # far beyond the linear segment
    linear = inbound_delay_draws(LEVELS, spread(5.0, 12.0), u)
    assert linear[-1] < x[-1]
    # altering a row that was not passed in cannot change the fit
    again = TailModel.fit(outcome[:50], top[:50], groups[:50])
    assert np.array_equal(again.for_group(3), tail.for_group(3))


def test_tail_model_changes_the_long_delay_probability():
    tail = np.linspace(1.0, 80.0, 200)
    with_tail = run(spread(0.0, 6.0), tail=tail, alts=[], seed=3, draws=4000)
    without = run(spread(0.0, 6.0), alts=[], seed=3, draws=4000)
    assert with_tail.band_probs[-1] >= without.band_probs[-1]
