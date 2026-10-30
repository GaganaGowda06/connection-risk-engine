from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from rail_risk.repay import DelayRepayRules, RepayError, expected_repay, load_rules

EDGES = (0, 15, 30, 60, 120)
ROOT = Path(__file__).parent.parent / "config"


def rules(**overrides) -> DelayRepayRules:
    base = {
        "name": "Test Rail",
        "status": "verified",
        "source_url": "https://example.test/delay-repay",
        "checked": "2026-10-15",
        "tiers": [
            {"from_min": 15, "to_min": 30, "single": 0.25, "return": 0.125},
            {"from_min": 30, "to_min": 60, "single": 0.5, "return": 0.25},
            {"from_min": 60, "to_min": 120, "single": 1.0, "return": 0.5},
            {"from_min": 120, "to_min": None, "single": 1.0, "return": 1.0},
        ],
    }
    base.update(overrides)
    unverified = {
        "name": "Other Rail",
        "status": "unverified",
        "source_url": "https://example.test/other",
        "checked": "2026-10-15",
    }
    return DelayRepayRules.model_validate({"operators": {"VT": base, "LM": unverified}})


P = [0.6, 0.2, 0.1, 0.07, 0.03]


def test_single_by_hand():
    # 0.2 * 12.50 + 0.1 * 25.00 + 0.07 * 50.00 + 0.03 * 50.00 = 2.5 + 2.5 + 3.5 + 1.5
    r = expected_repay(P, EDGES, "VT", "single", 50.0, rules())
    assert r.expected_payout == pytest.approx(10.0)
    assert [t.tier for t in r.tiers] == ["15-29", "30-59", "60-119", "120+"]
    assert [t.expected for t in r.tiers] == pytest.approx([2.5, 2.5, 3.5, 1.5])
    assert [t.payout_if_in_tier for t in r.tiers] == pytest.approx([12.5, 25.0, 50.0, 50.0])
    assert r.p_no_payout == pytest.approx(0.6)


def test_return_by_hand():
    # return share on an 80 pound return: 0.2*10 + 0.1*20 + 0.07*40 + 0.03*80 = 2+2+2.8+2.4
    r = expected_repay(P, EDGES, "VT", "return", 80.0, rules())
    assert r.expected_payout == pytest.approx(9.2)


def test_certain_outcomes_and_zero_fare():
    on_time = expected_repay([1, 0, 0, 0, 0], EDGES, "VT", "single", 50.0, rules())
    assert on_time.expected_payout == 0.0 and on_time.p_no_payout == 1.0
    sure_120 = expected_repay([0, 0, 0, 0, 1], EDGES, "VT", "return", 30.0, rules())
    assert sure_120.expected_payout == pytest.approx(30.0)
    assert expected_repay(P, EDGES, "VT", "single", 0.0, rules()).expected_payout == 0.0


def test_unverified_unknown_and_bad_inputs_are_errors_not_numbers():
    with pytest.raises(RepayError, match="unverified"):
        expected_repay(P, EDGES, "LM", "single", 50.0, rules())
    with pytest.raises(RepayError, match="no Delay Repay rules"):
        expected_repay(P, EDGES, "XX", "single", 50.0, rules())
    with pytest.raises(RepayError, match="fare"):
        expected_repay(P, EDGES, "VT", "single", -1.0, rules())
    with pytest.raises(RepayError, match="fare"):
        expected_repay(P, EDGES, "VT", "single", float("nan"), rules())
    with pytest.raises(RepayError, match="ticket type"):
        expected_repay(P, EDGES, "VT", "season", 50.0, rules())
    with pytest.raises(RepayError, match="sum to 1"):
        expected_repay([0.5, 0.2, 0.1, 0.1, 0.0], EDGES, "VT", "single", 50.0, rules())
    with pytest.raises(RepayError, match="band edge"):
        expected_repay([0.5, 0.5, 0, 0], (0, 20, 30, 60), "VT", "single", 50.0, rules())


def test_tiers_must_be_contiguous_ascending_and_non_overlapping():
    def tiers(*rows):
        return [{"from_min": a, "to_min": b, "single": s, "return": s / 2} for a, b, s in rows]

    with pytest.raises(ValidationError, match="not contiguous"):
        rules(tiers=tiers((15, 30, 0.25), (35, 60, 0.5), (60, None, 1.0)))
    with pytest.raises(ValidationError, match="overlap"):
        rules(tiers=tiers((15, 35, 0.25), (30, 60, 0.5), (60, None, 1.0)))
    with pytest.raises(ValidationError, match="open ended"):
        rules(tiers=tiers((15, 30, 0.25), (30, 60, 0.5)))
    with pytest.raises(ValidationError, match="positive range"):
        rules(tiers=tiers((30, 15, 0.25), (15, None, 1.0)))
    with pytest.raises(ValidationError, match="between 0 and 1"):
        rules(tiers=tiers((15, None, 1.5)))
    with pytest.raises(ValidationError, match="needs tiers"):
        rules(tiers=[])


def test_unverified_operator_cannot_carry_tiers():
    with pytest.raises(ValidationError, match="must not carry tiers"):
        DelayRepayRules.model_validate(
            {
                "operators": {
                    "LM": {
                        "name": "X",
                        "status": "unverified",
                        "source_url": "https://example.test",
                        "checked": "2026-10-15",
                        "tiers": [{"from_min": 15, "to_min": None, "single": 0.5, "return": 0.5}],
                    }
                }
            }
        )


def test_shipped_rules_match_what_was_read_from_the_operator_page():
    shipped = load_rules(ROOT / "delay_repay.yaml")
    vt = shipped.operator("VT")
    assert vt.status == "verified" and vt.claim_window_days == 28
    assert [(t.from_min, t.to_min, t.single, t.return_) for t in vt.tiers] == [
        (15, 30, 0.25, 0.125),
        (30, 60, 0.5, 0.25),
        (60, 120, 1.0, 0.5),
        (120, None, 1.0, 1.0),
    ]
    assert str(vt.checked) == "2026-10-15" and vt.source_url.startswith("https://www.avanti")
    lm = shipped.operators["LM"]  # recorded from the page, but not priced (shared TOC code)
    assert [(t.from_min, t.to_min, t.single, t.return_) for t in lm.tiers] == [
        (15, 30, 0.25, 0.125),
        (30, 60, 0.5, 0.25),
        (60, 120, 1.0, 0.5),
        (120, None, 1.0, 1.0),
    ]
    assert "London Northwestern" in lm.caveat  # the shared TOC code is flagged on every payout
    assert lm.shared_toc
    assert shipped.operators["LNR"].status == "unverified"
    with pytest.raises(RepayError, match="unverified"):
        shipped.operator("LNR")
    raw = yaml.safe_load((ROOT / "delay_repay.yaml").read_text())
    assert raw["operators"]["LNR"]["tiers"] == []


def test_expected_payout_is_linear_in_fare_and_ordered_by_ticket():
    a = expected_repay(P, EDGES, "VT", "single", 20.0, rules()).expected_payout
    b = expected_repay(P, EDGES, "VT", "single", 60.0, rules()).expected_payout
    assert b == pytest.approx(3 * a)
    ret = expected_repay(P, EDGES, "VT", "return", 20.0, rules()).expected_payout
    assert ret < a
    assert np.isfinite(ret)


def test_shipped_rules_price_avanti_but_never_a_shared_toc_or_an_unverified_operator():
    shipped = load_rules(ROOT / "delay_repay.yaml")
    assert expected_repay(P, EDGES, "VT", "single", 10.0, shipped).expected_payout > 0
    # TOC LM is West Midlands Railway and London Northwestern Railway: nothing is priced for it
    with pytest.raises(RepayError, match="not priced: rules unverified"):
        expected_repay(P, EDGES, "LM", "single", 10.0, shipped)
    with pytest.raises(RepayError, match="unverified"):  # London Northwestern by its own key
        expected_repay(P, EDGES, "LNR", "single", 10.0, shipped)
    with pytest.raises(RepayError, match="no Delay Repay rules"):  # an operator never recorded
        expected_repay(P, EDGES, "XC", "single", 10.0, shipped)
