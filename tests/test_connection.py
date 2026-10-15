import numpy as np
import pandas as pd
import pytest

from rail_risk.connection import (
    FixedDelays,
    OutboundDelays,
    Train,
    band_probabilities,
    hhmm_to_minutes,
    inbound_delay_draws,
    journey_repay,
    load_stations,
    minutes_to_hhmm,
    simulate,
)
from rail_risk.repay import RepayError, load_rules

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


def test_no_catchable_alternative_is_stranded_not_a_number():
    r = run(flat(90.0), alts=[])
    assert r.stranded and r.p_stranded == 1.0 and r.p_miss == 1.0
    assert r.delay_quantiles is None and r.band_probs is None and r.arrival_range() is None
    with pytest.raises(RepayError, match="stranded"):
        journey_repay(r, EDGES, "VT", "single", 50.0, load_rules("config/delay_repay.yaml"))
    # alternatives exist but all leave before the inbound can connect
    gone = run(flat(200.0))
    assert gone.stranded


def test_partly_stranded_reports_the_share_and_conditions_on_completion():
    q = spread(30.0, 40.0)  # the late end cannot reach the last alternative at 11:00
    r = run(q, alts=ALTS[:1])
    assert 0.0 < r.p_stranded < 1.0 and not r.stranded
    assert r.band_probs.sum() == pytest.approx(1.0)
    assert len(r.delays) == round((1 - r.p_stranded) * r.n_draws)


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
