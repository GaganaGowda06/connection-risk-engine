import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rail_risk.evaluation import (
    Bands,
    HistoricalBaseline,
    Predictions,
    SplitConfig,
    band_index,
    check_split,
    complete_days,
    evaluate,
    load_bands,
    load_split_config,
    make_split,
    persistence_baseline,
    score,
    timetable_baseline,
    write_reports,
)

BANDS = Bands((0, 15, 30, 60, 120), provisional=True)
CFG = SplitConfig(7, 7, 1, 16, 1000, 20261015, 0.95)
DAY0 = date(2026, 9, 1)


def days(n: int, start: date = DAY0) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


# split


def test_split_blocks_and_gaps():
    s = make_split(days(30), CFG)
    assert s.test == tuple(days(7, DAY0 + timedelta(days=23)))  # most recent 7 days
    assert s.validation == tuple(days(7, DAY0 + timedelta(days=15)))  # day 22 is the gap
    assert s.train == tuple(days(14))  # day 14 is the gap
    assert max(s.train) < min(s.validation) and max(s.validation) < min(s.test)
    assert len(set(s.train) | set(s.validation) | set(s.test)) == 28


def test_split_needs_enough_days():
    with pytest.raises(ValueError, match="need at least"):
        make_split(days(16), CFG)


def test_split_rejects_overlap_disorder_and_missing_gap():
    d = days(30)
    with pytest.raises(ValueError, match="two blocks"):
        check_split(make_split(d, CFG).__class__(tuple(d[:10]), tuple(d[9:15]), tuple(d[20:])))
    with pytest.raises(ValueError, match="time order"):
        check_split(make_split(d, CFG).__class__(tuple(d[15:20]), tuple(d[:5]), tuple(d[25:])))
    with pytest.raises(ValueError, match="gap"):  # d[9] and d[10] are adjacent days
        check_split(make_split(d, CFG).__class__(tuple(d[:10]), tuple(d[10:15]), tuple(d[20:])))
    with pytest.raises(ValueError, match="at least one day"):
        check_split(make_split(d, CFG).__class__((), tuple(d[11:15]), tuple(d[20:])))


def test_shipped_config_files_load():
    root = Path(__file__).parent.parent / "config"
    cfg = load_split_config(root / "split.yaml")
    assert (cfg.test_days, cfg.validation_days, cfg.gap_days, cfg.resamples) == (7, 7, 1, 1000)
    bands = load_bands(root / "delay_bands.yaml")
    assert bands.provisional is True
    assert bands.labels == ["0-14", "15-29", "30-59", "60-119", "120+"]


def test_complete_days_requires_all_windows_for_every_route(tmp_path):
    db = tmp_path / "m.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE windows (route TEXT, day TEXT, hour INTEGER)")
    for hour in range(16):
        con.execute("INSERT INTO windows VALUES ('a','2026-09-01',?)", (hour,))
        con.execute("INSERT INTO windows VALUES ('b','2026-09-01',?)", (hour,))
    for hour in range(16):
        con.execute("INSERT INTO windows VALUES ('a','2026-09-02',?)", (hour,))
    for hour in range(15):  # incomplete
        con.execute("INSERT INTO windows VALUES ('a','2026-09-03',?)", (hour,))
    con.commit()
    con.close()
    assert complete_days(db, ["a"], 16) == [date(2026, 9, 1), date(2026, 9, 2)]
    assert complete_days(db, ["a", "b"], 16) == [date(2026, 9, 1)]


# metrics on hand-computed cases


def test_band_index_edges_and_early_arrivals():
    got = band_index(np.array([-5.0, 0, 14.9, 15, 29.9, 30, 59, 60, 119, 120, 400]), BANDS)
    assert got.tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def one_date(y, point, lo=None, hi=None, probs=None, **kw):
    pred = Predictions(
        np.array(point, float),
        None if lo is None else np.array(lo, float),
        None if hi is None else np.array(hi, float),
        None if probs is None else np.array(probs, float),
    )
    return score(np.array([date(2026, 9, 1)] * len(y)), np.array(y, float), pred, BANDS, **kw)


def test_point_metrics_by_hand():
    r = one_date([0, 10, 20], [0, 12, 15], resamples=10)
    assert r["mae"]["value"] == pytest.approx((0 + 2 + 5) / 3)
    assert r["rmse"]["value"] == pytest.approx(np.sqrt((0 + 4 + 25) / 3))
    assert r["coverage"]["value"] is None and r["brier"]["value"] is None


def test_interval_metrics_by_hand():
    # the middle row misses: 10 is outside [11, 14]
    r = one_date([0, 10, 20], [0, 12, 15], lo=[-1, 11, 18], hi=[1, 14, 25], resamples=10)
    assert r["coverage"]["value"] == pytest.approx(2 / 3)
    assert r["width"]["value"] == pytest.approx((2 + 3 + 7) / 3)


def test_band_metrics_by_hand():
    # truth bands: 0 -> band 0, 20 -> band 1
    probs = [[1, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0]]
    r = one_date([0, 20], [0, 0], probs=probs, resamples=10)
    # row 1 perfect: brier 0, log loss 0. row 2: (0.5-0)^2 + (0.5-1)^2 = 0.5, log loss -ln 0.5
    assert r["brier"]["value"] == pytest.approx(0.5 / 2)
    assert r["log_loss"]["value"] == pytest.approx(-np.log(0.5) / 2)


def test_log_loss_is_clipped_for_zero_probability():
    r = one_date([20], [0], probs=[[1, 0, 0, 0, 0]], resamples=10)
    assert r["log_loss"]["value"] == pytest.approx(-np.log(1e-15))
    assert r["brier"]["value"] == pytest.approx(2.0)


# bootstrap


def multi_date_inputs():
    rng = np.random.default_rng(0)
    dates = np.repeat([date(2026, 9, 1) + timedelta(days=i) for i in range(6)], 20)
    y = rng.normal(5, 4, size=len(dates))
    return dates, y, Predictions(np.zeros(len(y)))


def test_bootstrap_is_reproducible_under_the_fixed_seed():
    dates, y, pred = multi_date_inputs()
    a = score(dates, y, pred, BANDS, resamples=1000, seed=7)
    b = score(dates, y, pred, BANDS, resamples=1000, seed=7)
    c = score(dates, y, pred, BANDS, resamples=1000, seed=8)
    assert a == b
    assert a["mae"]["ci_low"] != c["mae"]["ci_low"]
    assert a["mae"]["ci_low"] <= a["mae"]["value"] <= a["mae"]["ci_high"]


def test_bootstrap_resamples_dates_not_rows():
    # two dates with MAE 0 and 10. Resampling dates can only give MAE 0, 5 or 10, so the CI
    # reaches the extremes. Resampling the 100 rows would give a much tighter interval.
    dates = np.array([date(2026, 9, 1)] * 50 + [date(2026, 9, 2)] * 50)
    y = np.array([0.0] * 50 + [10.0] * 50)
    r = score(dates, y, Predictions(np.zeros(100)), BANDS, resamples=1000, seed=1)
    assert r["mae"]["value"] == pytest.approx(5.0)
    assert r["mae"]["ci_low"] == pytest.approx(0.0)
    assert r["mae"]["ci_high"] == pytest.approx(10.0)


# baselines and the no-leakage property


def snapshot_frame(n_days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for i, d in enumerate(days(n_days)):
        for k in range(6):
            label = float(rng.gamma(2.0, 3.0))
            for h in (120, 60, 15):
                rows.append(
                    {
                        "rid": f"{d:%Y%m%d}{k:07d}",
                        "route_id": "r1",
                        "service_date": d,
                        "horizon_min": h,
                        "dest_hour": 10 + k % 2,
                        "last_stop_lateness_min": None if h == 120 else label * 0.8,
                        "hist_median_28d": None if i == 0 else 5.0 + (k % 2),
                        "label_delay_min": label,
                    }
                )
    return pd.DataFrame(rows)


def test_timetable_and_persistence_baselines():
    df = pd.DataFrame({"last_stop_lateness_min": [None, 20.0, -2.0]})
    t = timetable_baseline(df, BANDS)
    assert t.point.tolist() == [0, 0, 0] and t.probs[:, 0].tolist() == [1, 1, 1]
    p = persistence_baseline(df, BANDS)
    assert p.point.tolist() == [0.0, 20.0, -2.0]
    assert p.probs.argmax(axis=1).tolist() == [0, 1, 0]
    assert t.lo is None and p.lo is None


def test_historical_fit_ignores_validation_and_test_rows():
    df = snapshot_frame()
    s = make_split(sorted(df["service_date"].unique()), CFG)
    train = df[df["service_date"].isin(s.train)]
    base = HistoricalBaseline(BANDS).fit(train)
    altered = df.copy()
    held_out = altered["service_date"].isin(s.validation + s.test)
    altered.loc[held_out, "label_delay_min"] += 500.0
    again = HistoricalBaseline(BANDS).fit(altered[altered["service_date"].isin(s.train)])
    assert base.q_lo == again.q_lo and base.q_hi == again.q_hi and base.median == again.median
    # and the end-to-end evaluate() only ever fits on the train block
    r1 = evaluate(df, s, BANDS, CFG)
    r2 = evaluate(altered.assign(label_delay_min=altered["label_delay_min"]), s, BANDS, CFG)
    assert (
        r1["results"]["historical"]["validation"]["60"]["width"]
        == (r2["results"]["historical"]["validation"]["60"]["width"])
    )


def test_historical_interval_uses_residual_quantiles_and_backs_off():
    df = snapshot_frame()
    fit = HistoricalBaseline(BANDS).fit(df)
    new_route = df.head(3).assign(route_id="unseen", dest_hour=7, hist_median_28d=None)
    p = fit.predict(new_route)  # unseen route falls back to the global fit
    assert np.isfinite(p.point).all() and (p.hi > p.lo).all()
    assert p.probs.sum(axis=1) == pytest.approx(1.0)
    assert (p.probs > 0).all()  # smoothing keeps every band possible


def test_reports_are_written_without_em_dashes(tmp_path):
    df = snapshot_frame()
    s = make_split(sorted(df["service_date"].unique()), CFG)
    result = evaluate(df, s, BANDS, CFG)
    write_reports(result, tmp_path)
    md = (tmp_path / "baselines.md").read_text()
    assert "—" not in md and "bar to beat" in md and "## T-60 minutes" in md
    assert (tmp_path / "baselines.json").exists()
    assert "provisional" in md
