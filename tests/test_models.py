from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from rail_risk.evaluation import Bands, SplitConfig, make_split
from rail_risk.features.snapshots import FEATURE_COLUMNS
from rail_risk.lateness import group_labels, lateness_base, lateness_groups, merge_groups
from rail_risk.models.bands import IsotonicOvR, TemperatureScaler, choose_calibrator
from rail_risk.models.pipeline import fit_all, guard, run_training, validation_halves
from rail_risk.models.quantile import (
    QuantileFit,
    apply_offset,
    conformal_offset,
    fit_models,
    fit_quantile,
    sort_quantiles,
)

BANDS = Bands((0, 15, 30, 60, 120), provisional=True)
EVAL_CFG = SplitConfig(7, 7, 1, 16, 50, 1, 0.95)
CFG = {
    "quantiles": [0.1, 0.5, 0.9],
    "nominal_coverage": 0.8,
    "split_validation_min_days": 10,
    "seed": 11,
    "num_threads": 1,
    "max_rounds": 120,
    "early_stopping_rounds": 15,
    "grid": {
        "num_leaves": [4, 8],
        "min_data_in_leaf": [20],
        "learning_rate": [0.1],
        "feature_fraction": [0.8],
    },  # fmt: skip
    "fixed": {"lambda_l2": 10.0, "max_depth": 5, "bagging_fraction": 0.8, "bagging_freq": 1},
    "band_model": {
        "num_leaves": 4,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
    },  # fmt: skip
}


def synthetic(n: int, seed: int, day0: date = date(2026, 1, 1)) -> pd.DataFrame:
    """Heteroscedastic delays with a known distribution: y = 3 x1 + (1 + 2 x2) * noise."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(0, 4, n), rng.uniform(0, 1, n)
    y = 3 * x1 + (1 + 2 * x2) * rng.normal(0, 2, n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "horizon_min": 60,
            "label_delay_min": y,
            "service_date": [day0 + timedelta(days=int(i % 20)) for i in range(n)],
        }
    )


def test_quantiles_never_cross_even_when_models_disagree():
    raw = np.array([[5.0, 2.0, 9.0], [1.0, 3.0, 2.0], [4.0, 4.0, 4.0]])
    out = sort_quantiles(raw)
    assert (out[:, 0] <= out[:, 1]).all() and (out[:, 1] <= out[:, 2]).all()
    lo, hi = apply_offset(out[:, 0], out[:, 1], out[:, 2], offset=-50.0)  # huge narrowing
    assert (lo <= out[:, 1]).all() and (out[:, 1] <= hi).all()
    lo, hi = apply_offset(out[:, 0], out[:, 1], out[:, 2], offset=2.0)
    assert (hi - lo >= out[:, 2] - out[:, 0] + 4.0 - 1e-9).all()


def test_conformal_offset_by_hand():
    y = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    lo, hi = y - 1, y + 1
    assert conformal_offset(y, lo, hi, 0.8) == pytest.approx(-1.0)  # intervals already wide
    y_out = y + np.array([0, 0, 0, 5, 9.0])  # two misses by 4 and 8 beyond the band
    assert conformal_offset(y_out, lo, hi, 0.8) == pytest.approx(8.0)


def test_conformal_coverage_on_known_distribution():
    df = synthetic(10_000, seed=5)
    train, calib, test = df.iloc[:4000], df.iloc[4000:7000], df.iloc[7000:]
    cols = ["x1", "x2", "horizon_min"]
    fit, grid = fit_quantile(train, calib, calib, CFG, columns=cols)
    assert len(grid) == 2
    _, lo, hi = fit.predict(test)
    y = test["label_delay_min"].to_numpy()
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    assert abs(coverage - 0.8) < 0.02
    assert float(np.mean(hi - lo)) > 0
    # the offset is per horizon and the median is the point forecast
    assert set(fit.global_offsets) == {60}
    assert set(fit.offsets) == {(60, 0)}  # no lateness column: one group


def test_calibrated_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5), size=400)
    y = np.array([rng.choice(5, p=row) for row in p])
    dates = np.array([date(2026, 9, 1) + timedelta(days=i % 8) for i in range(400)])
    for cal in (IsotonicOvR().fit(p, y), TemperatureScaler().fit(p, y)):
        out = cal.transform(p)
        assert out.shape == p.shape
        assert out.sum(axis=1) == pytest.approx(1.0)
        assert (out >= 0).all() and (out <= 1).all()
    chosen, cv = choose_calibrator(p, y, dates)
    assert chosen.name in {"isotonic", "temperature"} and set(cv) == {"isotonic", "temperature"}
    assert chosen.transform(p).sum(axis=1) == pytest.approx(1.0)


def test_guard_threshold():
    assert guard(59, 60) == "smoke"
    assert guard(60, 60) == "reportable"
    assert guard(120, 60) == "reportable"


def test_validation_is_split_in_two_only_with_enough_days():
    few = synthetic(700, 1).assign(service_date=lambda d: d["service_date"].where(
        d["service_date"] < date(2026, 1, 8), date(2026, 1, 7)))  # fmt: skip
    tune, cal, halved = validation_halves(few, 10)
    assert halved is False and len(tune) == len(cal) == len(few)
    many = synthetic(800, 1)  # 20 days
    tune, cal, halved = validation_halves(many, 10)
    assert halved is True
    assert max(tune["service_date"]) < min(cal["service_date"])
    assert len(tune) + len(cal) == len(many)


def snapshots(days: int = 26, per_day: int = 24, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2026, 8, 1)
    for i in range(days):
        d = start + timedelta(days=i)
        for k in range(per_day):
            base = rng.gamma(2.0, 3.0)
            for h in (120, 60, 15):
                row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
                row.update(
                    {
                        "rid": f"{d:%Y%m%d}{k:07d}",
                        "route_id": "r1",
                        "service_date": d,
                        "horizon_min": h,
                        "toc_code": "VT" if k % 2 else "LM",
                        "dest_hour": 9 + k % 3,
                        "is_bank_holiday": False,
                        "last_stop_lateness_min": None if h == 120 else base * 0.7,
                        "hist_median_28d": None if i < 2 else 3.0,
                        "label_delay_min": base,
                        "is_candidate_cancelled": False,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_run_is_deterministic_and_blind_to_test_labels():
    df = snapshots()
    split = make_split(sorted(df["service_date"].unique()), EVAL_CFG)
    a, fit_a = run_training(df, split, BANDS, EVAL_CFG, CFG)
    b, fit_b = run_training(df, split, BANDS, EVAL_CFG, CFG)
    assert a["model"] == b["model"]
    rows = df[df["service_date"].isin(split.validation)]
    assert (fit_a.quantile.predict(rows)[1] == fit_b.quantile.predict(rows)[1]).all()

    # change only the test labels: nothing fitted or tuned may move
    altered = df.copy()
    altered.loc[altered["service_date"].isin(split.test), "label_delay_min"] += 300.0
    c, fit_c = run_training(altered, split, BANDS, EVAL_CFG, CFG)
    for key in ("params", "best_iteration", "conformal_offsets", "global_offsets",
                "target_by_horizon", "group_merges", "calibration_comparison",
                "band_calibration", "band_calibration_cv_log_loss", "grid"):  # fmt: skip
        assert c["model"][key] == a["model"][key], key
    assert (fit_c.quantile.predict(rows)[1] == fit_a.quantile.predict(rows)[1]).all()
    assert c["results"]["model"]["validation"] == a["results"]["model"]["validation"]
    assert c["results"]["model"]["test"] != a["results"]["model"]["test"]  # but test scores move


def test_fit_all_returns_models_for_every_quantile_and_horizon():
    df = snapshots(days=14)
    train = df[df["service_date"] < date(2026, 8, 9)]
    valid = df[df["service_date"] >= date(2026, 8, 9)]
    fitted = fit_all(train, valid, BANDS, CFG)
    for fit in fitted.quantile.fits.values():
        assert set(fit.boosters) == {0.1, 0.5, 0.9}
        assert set(fit.global_offsets) == {120, 60, 15}
        assert {h for h, _ in fit.offsets} == {120, 60, 15}
    assert set(fitted.quantile.choice) == {120, 60, 15}
    assert set(fitted.quantile.choice.values()) <= {"direct", "residual"}
    probs = fitted.bands.predict(valid)
    assert probs.shape == (len(valid), 5) and probs.sum(axis=1) == pytest.approx(1.0)


def test_small_groups_merge_into_neighbour_and_are_logged():
    mapping, merges = merge_groups({0: 200, 1: 400, 2: 12, 3: 5}, 30)
    # 3 (n=5) joins 2 (n=12), then the merged 14 rows join the smaller neighbour, 0 or 1
    assert mapping[2] == mapping[3]
    assert merges[0] == ([3], [2])
    assert len(merges) == 2 and merges[1][0] == [2, 3]
    assert set(mapping.values()) == {0, 1}
    assert mapping[0] == 0 and mapping[1] == 1
    # nothing small: nothing merges
    assert merge_groups({0: 50, 1: 60, 2: 31, 3: 30}, 30) == ({0: 0, 1: 1, 2: 2, 3: 3}, [])
    # everything small: collapses to one group rather than looping
    mapping, _ = merge_groups({0: 3, 1: 4, 2: 5, 3: 6}, 30)
    assert set(mapping.values()) == {0}


def test_lateness_groups_use_only_lateness_at_t():
    df = pd.DataFrame({"last_stop_lateness_min": [None, -3.0, 0.0, 4.9, 5.0, 14.9, 15.0, 90.0]})
    assert lateness_groups(df).tolist() == [0, 1, 1, 1, 2, 2, 3, 3]
    assert group_labels() == ["not_started", "0-4", "5-14", "15+"]
    assert "label_delay_min" not in df.columns  # the function needs no label


def hetero(n: int, seed: int) -> pd.DataFrame:
    """Noise scale grows with lateness at T, so a global offset under-covers the late group."""
    rng = np.random.default_rng(seed)
    late = np.where(rng.random(n) < 0.2, np.nan, rng.gamma(1.5, 6.0, n))
    base = np.nan_to_num(late)
    scale = np.where(np.isnan(late), 1.0, 0.5 + 0.35 * base)
    y = base + rng.normal(0, 1, n) * scale
    return pd.DataFrame(
        {
            "last_stop_lateness_min": late,
            "horizon_min": 60,
            "label_delay_min": y,
            "service_date": [date(2026, 1, 1) + timedelta(days=int(i % 10)) for i in range(n)],
        }
    )


def test_mondrian_coverage_is_nominal_within_each_group():
    df = hetero(20_000 * 3, seed=21)
    train, calib, test = df.iloc[:20_000], df.iloc[20_000:40_000], df.iloc[40_000:]
    # the model is blind to lateness, so only the groupwise offsets can fix group coverage
    fit, _ = fit_quantile(train, calib, calib, CFG, columns=["horizon_min"])
    groups = lateness_groups(test)
    y = test["label_delay_min"].to_numpy()
    for mode in ("mondrian",):
        _, lo, hi = fit.predict(test, mode)
        for k in range(4):
            m = groups == k
            assert m.sum() > 500
            assert abs(float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))) - 0.8) < 0.03, (mode, k)
    # a single global offset is visibly off in at least one group on this data
    _, glo, ghi = fit.predict(test, "global")
    gaps = [
        abs(
            float(
                np.mean((y[groups == k] >= glo[groups == k]) & (y[groups == k] <= ghi[groups == k]))
            )
            - 0.8
        )
        for k in range(4)
    ]
    assert max(gaps) > 0.03


def test_residual_target_reconstructs_delay_scale():
    df = hetero(6_000, seed=3)
    cols = ["last_stop_lateness_min", "horizon_min"]
    train, calib = df.iloc[:4000], df.iloc[4000:]
    fit, _ = fit_quantile(train, calib, calib, CFG, columns=cols, target="residual")
    assert isinstance(fit, QuantileFit) and fit.target == "residual"
    x = fit.design.transform(calib)
    stored = np.column_stack([fit.boosters[q].predict(x) for q in fit.quantiles])
    expected = np.sort(stored + lateness_base(calib)[:, None], axis=1)
    assert fit.raw(calib) == pytest.approx(expected)
    # not started falls back to a base of zero, so direct and residual agree on that scale
    started = calib["last_stop_lateness_min"].notna().to_numpy()
    assert (lateness_base(calib)[~started] == 0).all()
    point, lo, hi = fit.predict(calib)
    assert (lo <= point).all() and (point <= hi).all()


def test_fit_models_keeps_the_better_target_per_horizon():
    df = hetero(6_000, seed=8)  # y is close to lateness, so residual should win
    cols = ["last_stop_lateness_min", "horizon_min"]
    model, grids = fit_models(df.iloc[:4000], df.iloc[4000:], df.iloc[4000:], CFG, cols)
    assert set(grids) == {"direct", "residual"}
    assert model.choice[60] == min(model.tune_pinball, key=lambda t: model.tune_pinball[t][60])
