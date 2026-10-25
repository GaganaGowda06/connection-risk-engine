"""Adaptive calibration: no outcome may be used before its train has arrived."""

import numpy as np
import pandas as pd

from rail_risk.network_model import adaptive as A


def rows(n: int = 600, seed: int = 0, routes=("a", "b")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-10-01 06:00", tz="UTC")
    parts = []
    for r in routes:
        pt = t0 + pd.to_timedelta(np.sort(rng.integers(0, 14 * 24 * 60, n)), unit="m")
        y = rng.normal(0, 3, n)
        point = np.zeros(n)
        parts.append(
            pd.DataFrame(
                {
                    "route": r,
                    "horizon_min": 15,
                    "pt_arr": pt,
                    "snap_time": pt - pd.Timedelta(minutes=15),
                    "target_min": y,
                    "raw_lo": point - 2.0,
                    "raw_hi": point + 2.0,
                    "ssd": pt.date,
                    "late_now_min": 0.0,
                    "sched_duration_min": 60.0,
                    "stops_remaining": 5,
                }
            )  # fmt: skip
        )
    return pd.concat(parts, ignore_index=True)


def scores() -> A.GlobalScores:
    df = rows(4000, seed=9)
    df["lo"], df["hi"] = df["raw_lo"], df["raw_hi"]
    return A.GlobalScores.from_pool(df)


def test_a_row_never_sees_an_outcome_that_arrives_after_its_snapshot_time():
    gs = scores()
    base = rows()
    for method in A.METHODS:
        ref = A.run(base, gs, method, 0.02, min_window=20)
        for i in (50, 200, 450, 700):
            ti = base.loc[i, "snap_time"]
            arrival = base["pt_arr"] + pd.to_timedelta(base["target_min"], unit="m")
            future = (arrival > ti) & (base.index != i)
            changed = base.copy()
            changed.loc[future, "target_min"] = changed.loc[future, "target_min"] * 7 + 40
            out = A.run(changed, gs, method, 0.02, min_window=20)
            assert out.loc[i, "lo"] == ref.loc[i, "lo"], (method, i)
            assert out.loc[i, "hi"] == ref.loc[i, "hi"], (method, i)


def test_every_route_starts_from_the_global_calibration_and_adapts_after_misses():
    gs = scores()
    df = rows(300)
    out = A.run(df, gs, "aci_route", 0.05)
    first = df.groupby("route").head(1).index
    q = gs.q(15, 1 - A.TARGET_MISS)
    w = (df["raw_hi"] - df["raw_lo"]).loc[first]
    assert np.allclose(out.loc[first, "hi"] - df.loc[first, "raw_hi"], q * w)
    wide = df.copy()
    wide["target_min"] = wide["target_min"] * 4  # the route suddenly runs far from the forecast
    adapted = A.run(wide, gs, "aci_route", 0.05)
    w_end = (
        (adapted["hi"] - adapted["lo"]).groupby(wide["route"]).apply(lambda s: s.iloc[-50:].mean())
    )
    w_start = (
        (adapted["hi"] - adapted["lo"]).groupby(wide["route"]).apply(lambda s: s.iloc[:5].mean())
    )
    assert (w_end > 1.5 * w_start).all()


def test_a_silent_route_keeps_its_state_and_other_routes_are_unaffected():
    gs = scores()
    df = rows(400)
    silent = df[
        ~(
            (df["route"] == "a")
            & df["snap_time"].between(
                pd.Timestamp("2026-10-05", tz="UTC"), pd.Timestamp("2026-10-08", tz="UTC")
            )
        )
    ]
    full = A.run(df, gs, "aci_route", 0.02)
    cut = A.run(silent, gs, "aci_route", 0.02)
    b = df.index[df["route"] == "b"]
    assert np.array_equal(full.loc[b, "lo"], cut.loc[b, "lo"])
    assert np.isfinite(cut[["lo", "hi"]]).all().all()


def test_microsecond_timestamps_are_read_as_seconds():
    t = pd.Series(pd.to_datetime(["2026-10-01 06:00"], utc=True)).astype("datetime64[us, UTC]")
    ns = t.astype("datetime64[ns, UTC]")
    assert (
        A._seconds(t)[0]
        == A._seconds(ns)[0]
        == pd.Timestamp("2026-10-01 06:00", tz="UTC").timestamp()
    )
