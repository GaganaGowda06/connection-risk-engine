"""Recalibration of P(miss): choice on the tune half, serialisation, and use in /connection."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rail_risk import pmiss
from rail_risk.models.bands import _METHODS
from rail_risk.pmiss import PMissCalibration, fit_pmiss_calibration
from test_connection_api import INBOUND, client, connect, local, make, world
from world import ts


def rows(n: int, seed: int, over: float):
    """Predictions that over-predict by the factor `over`, over 10 dates."""
    rng = np.random.default_rng(seed)
    truth = rng.beta(0.7, 5.0, n)
    y = (rng.random(n) < truth).astype(float)
    p = np.clip(truth * over, 0, 1)
    dates = np.array([date(2026, 9, 24) + timedelta(days=i % 10) for i in range(n)])
    return p, y, dates


def test_an_overpredicting_forecast_gets_a_map_that_removes_the_bias():
    p, y, dates = rows(6000, 1, over=2.0)
    cal = fit_pmiss_calibration(p, y, dates)
    assert cal.method in {"platt", "isotonic"} and set(cal.tune_log_loss) == {
        "none",
        "platt",
        "isotonic",
    }
    assert cal.tune_log_loss[cal.method] < cal.tune_log_loss["none"]
    assert abs(cal.apply(p).mean() - y.mean()) < abs(p.mean() - y.mean())
    assert cal.n_tune > 0 and cal.n_calibrate > 0 and cal.n_tune + cal.n_calibrate == len(p)


def test_a_calibrated_forecast_keeps_none_and_ties_go_to_the_simpler_method():
    p, y, dates = rows(4000, 2, over=1.0)
    assert fit_pmiss_calibration(p, y, dates).method == "none"
    tiny = fit_pmiss_calibration(p[:20], y[:20], dates[:20])  # too few rows to compare
    assert tiny.method == "none" and tiny.apply(0.3)[0] == pytest.approx(0.3)
    one_day = fit_pmiss_calibration(p, y, np.full(len(p), date(2026, 9, 24)))
    assert one_day.method == "none"


def test_the_fit_uses_the_earlier_dates_to_tune_and_the_later_to_calibrate():
    p, y, dates = rows(3000, 3, over=2.0)
    a = fit_pmiss_calibration(p, y, dates)
    flipped = y.copy()
    early = dates <= sorted(set(dates.tolist()))[4]
    flipped[early] = 1.0 - flipped[early]  # only tune rows change
    b = fit_pmiss_calibration(p, flipped, dates)
    assert a.params != b.params or a.tune_log_loss != b.tune_log_loss
    late = ~early
    assert a.n_calibrate == int(late.sum()) and a.n_tune == int(early.sum())


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_round_trip_through_json_keeps_the_map(method, tmp_path):
    p, y, dates = rows(5000, 4, over=2.5)
    fitted = _METHODS[method]().fit(p, y)
    cal = PMissCalibration(method, pmiss._params(method, fitted))
    assert np.allclose(cal.apply(np.linspace(0, 1, 11)), fitted.predict(np.linspace(0, 1, 11)))
    path = tmp_path / "c.json"
    pmiss.save(cal, path)
    again = pmiss.load(path)
    grid = np.linspace(0, 1, 11)
    assert again.method == method and np.allclose(again.apply(grid), cal.apply(grid))
    assert ((again.apply(grid) >= 0) & (again.apply(grid) <= 1)).all()


def test_the_map_is_skipped_for_arrived_inbounds_in_the_predictions():
    pred = pd.DataFrame(
        {"p_miss": [0.4, 0.4, 0.0], "state": ["approaching", "due_or_late", "arrived"]}
    )
    cal = PMissCalibration("platt", {"coef": 1.0, "intercept": -1.0})
    out = pmiss.apply_to_predictions(pred, cal)
    assert out["p_miss_cal"].iloc[2] == 0.0 and out["p_miss_cal"].iloc[0] < 0.4
    assert out["p_miss_cal"].iloc[0] == out["p_miss_cal"].iloc[1]


def test_the_fitting_set_is_validation_pairs_with_an_outcome_and_no_arrived_inbound():
    pred = pd.DataFrame(
        {
            "block": ["validation"] * 4 + ["test"],
            "missed": [0.0, 1.0, np.nan, 0.0, 1.0],
            "state": ["approaching", "due_or_late", "approaching", "arrived", "approaching"],
        }
    )
    assert len(pmiss.split_frame(pred)) == 2


def test_connection_endpoint_applies_the_calibration_and_the_alert_follows_it(
    tmp_path, features, bundle_pair
):
    service = make(tmp_path, features, bundle_pair)
    with client(service) as c:
        raw = connect(c, outbound_departure=local("10:47")).json()
        service.pmiss_calibration = PMissCalibration("platt", {"coef": 1.0, "intercept": -1.0})
        cal = connect(c, outbound_departure=local("10:47")).json()
    assert raw["p_miss"] > cal["p_miss"] > 0.0 or raw["p_miss"] == 0.0
    assert any("recalibrated (platt)" in n for n in cal["data_quality"]["notes"])
    assert not any("recalibrated" in n for n in raw["data_quality"]["notes"])
    sens = {s["minutes"]: s["p_miss"] for s in cal["sensitivity"] if s["feasible"]}
    assert sens[10] == cal["p_miss"]  # the table and the headline agree


def test_an_arrived_inbound_is_not_recalibrated(tmp_path, features, bundle_pair):
    ev = world(inbound_actual="12:37")
    service = make(tmp_path, features, bundle_pair, events=ev, newest="12:30",
                   now=ts("12:40").to_pydatetime())  # fmt: skip
    service.pmiss_calibration = PMissCalibration("platt", {"coef": 1.0, "intercept": -5.0})
    with client(service) as c:
        body = connect(c, outbound_departure=local("10:47")).json()
    assert body["inbound"]["state"] == "arrived" and body["p_miss"] == 1.0
    assert not any("recalibrated" in n for n in body["data_quality"]["notes"])


def test_the_committed_calibration_file_loads_and_is_a_valid_map():
    cal = pmiss.load(Path("config/pmiss_calibration.json"))
    grid = np.linspace(0, 1, 21)
    out = cal.apply(grid)
    assert cal.method in {"none", "platt", "isotonic"} and ((out >= 0) & (out <= 1)).all()
    assert (np.diff(out) >= -1e-12).all()  # a probability map never reverses the order
    assert INBOUND  # imported world is the same one the API tests use
