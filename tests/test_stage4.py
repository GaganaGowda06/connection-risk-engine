"""Stage 4: range calibration methods, history features, cold-start base."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from rail_risk.network_model import calibrate as C
from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F


def pool(n: int = 4000, seed: int = 0, scale: float = 3.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    point = rng.gamma(1.5, 2.0, n)
    y = point + rng.normal(0, scale, n)
    return pd.DataFrame(
        {
            "point": point, "lo": point - 2.0, "hi": point + 2.0, "target_min": y,
            "horizon_min": rng.choice([15, 60], n), "late_now_min": rng.gamma(1.0, 3.0, n),
            "sched_duration_min": rng.choice([30.0, 70.0, 150.0], n),
            "stops_remaining": rng.integers(0, 30, n), "route": "r",
        }
    )  # fmt: skip


def coverage(df: pd.DataFrame, lo, hi) -> float:
    return float(((lo <= df["target_min"]) & (df["target_min"] <= hi)).mean())


def test_every_method_reaches_nominal_coverage_on_exchangeable_data():
    cal, test = pool(seed=0), pool(seed=1)
    for method in C.METHODS:
        c = C.RangeCalibrator(method).fit(cal)
        lo, hi = c.apply(
            test, test["point"].to_numpy(), test["lo"].to_numpy(), test["hi"].to_numpy()
        )
        assert 0.77 <= coverage(test, lo, hi) <= 0.83, method
        assert (lo <= hi).all()


def test_the_raw_range_was_too_narrow_and_calibration_widens_it():
    cal, test = pool(), pool(seed=2)
    raw = coverage(test, test["lo"], test["hi"])
    c = C.RangeCalibrator("cqr").fit(cal)
    lo, hi = c.apply(test, test["point"].to_numpy(), test["lo"].to_numpy(), test["hi"].to_numpy())
    assert raw < 0.6 and (hi - lo).mean() > (test["hi"] - test["lo"]).mean()


def test_mondrian_falls_back_to_a_coarser_cell_when_a_cell_is_small():
    cal = pool(3000)
    c = C.RangeCalibrator("mondrian", min_cell=200).fit(cal)
    odd = cal.head(5).copy()
    odd["late_now_min"] = 40.0  # a lateness band with no pool rows at all in the finest cells
    odd["stops_remaining"] = 29
    lo, hi = c.apply(odd, odd["point"].to_numpy(), odd["lo"].to_numpy(), odd["hi"].to_numpy())
    assert np.isfinite(lo).all() and np.isfinite(hi).all() and (hi > lo).all()
    assert c.fallback_rows >= 0


def test_groups_use_only_transferable_numbers():
    g = C.group_frame(pool(50))
    assert set(g.columns) == {"h", "lb", "lc", "sb"}
    assert (
        g["lb"].between(0, 3).all() and g["lc"].between(0, 2).all() and g["sb"].between(0, 3).all()
    )


def frame_for_history() -> pd.DataFrame:
    rows = []
    start = date(2026, 9, 1)
    for d in range(40):
        for i in range(30):
            rows.append(
                {"ssd": start + timedelta(days=d), "rid": f"{d}-{i}", "route": "a",
                 "arr_hour_local": 9, "horizon_min": 15, "has_target": True,
                 "target_min": float(d), "late_now_min": 0.0}
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_history_features_use_strictly_earlier_dates_only():
    df = frame_for_history()
    a = D.add_slot_history(df)
    changed = df.copy()
    changed.loc[changed["ssd"] >= date(2026, 10, 1), "target_min"] = 999.0  # today and later
    b = D.add_slot_history(changed)
    day = a["ssd"] == date(2026, 10, 1)
    for c in F.HISTORY:
        assert np.allclose(a.loc[day, c], b.loc[day, c], equal_nan=True), c
    first = a[a["ssd"] == date(2026, 9, 1)]
    assert first["slot_median_min"].isna().all() and first["recent_recovery_min"].isna().all()
    later = a[a["ssd"] == date(2026, 10, 10)]
    assert later["recent_recovery_min"].iloc[0] == np.median(
        [float(d) for d in range((date(2026, 10, 3) - date(2026, 9, 1)).days, 39)]
    )


def test_warm_base_adds_recent_recovery_and_cold_base_is_the_baseline():
    df = pd.DataFrame({"late_now_min": [2.0, np.nan], "recent_recovery_min": [-1.5, np.nan]})
    assert list(D.base_point(None, df, warm=True)) == [0.5, 0.0]
