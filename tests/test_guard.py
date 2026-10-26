import json

import numpy as np
import pandas as pd
import pytest

from rail_risk.ingest import pport_extract as PX
from rail_risk.serving.engine import Engine
from rail_risk.serving.guard import Guard


def frame(age, gap60=0.0, gap240=0.0):
    n = len(age)
    return pd.DataFrame(
        {
            "feed_age_min": age, "feed_gap60": gap60, "feed_gap240": gap240, "feed_gap120": gap60,
            "p_b": 0.9, "p_bar": 0.1, "point": 5.0, "aci_lo": 0.0, "aci_hi": 10.0,
            "static_lo": -5.0, "static_hi": 15.0, "p_late": 0.8, "hist_p": 0.2,
        },
        index=range(n),
    )  # fmt: skip


BLEND = Guard(
    design="blend", flag_window=240, pair_w=[-0.4, 3.0, -2.0, -2.0], late_w=[-0.1, -0.4, -1.0, 1.0],
    range_w=[0.6, 0.0, 0.0],
)  # fmt: skip


def test_fresh_feed_serves_the_model_in_every_design():
    df = frame(np.array([0.0, 3.0, 12.0]), 5.0, 5.0)
    for g in (
        Guard(design="none"),
        Guard(design="binary"),
        Guard(design="widen", widen_k=2),
        BLEND,
    ):
        r, a = g.pairs(df), g.arrivals(df)
        assert np.allclose(r.p_final, 0.9) and not r.stale.any()
        assert np.allclose(a.final_lo, 0.0) and np.allclose(a.p_late_final, 0.8)
        assert (r.stale_reason == "").all()


def test_binary_falls_back_and_says_why():
    df = frame(np.array([40.0]), 40.0, 40.0)
    r, a = Guard(design="binary").pairs(df), Guard(design="binary").arrivals(df)
    assert (
        r.p_final[0] == pytest.approx(0.1) and r.stale[0] and "feed silence 40" in r.stale_reason[0]
    )
    assert a.final_lo[0] == -5.0 and a.p_late_final[0] == pytest.approx(0.2)


def test_widen_keeps_model_and_widens_by_silence():
    g = Guard(design="widen", widen_k=1.0)
    df = frame(np.array([60.0]), 60.0, 60.0)
    r, a = g.pairs(df), g.arrivals(df)
    assert r.p_final[0] == 0.9 and r.stale[0]
    assert a.final_lo[0] == pytest.approx(5 - 2 * 5) and a.final_hi[0] == pytest.approx(5 + 2 * 5)


def test_blend_is_between_model_and_fallback_when_stale():
    df = frame(np.array([90.0]), 90.0, 90.0)
    r = BLEND.pairs(df)
    assert r.stale[0] and 0.1 <= r.p_final[0] <= 0.9 and 0 < r.fallback_w[0] <= 1


def test_guard_round_trips_through_json():
    assert Guard.from_json(json.loads(json.dumps(BLEND.to_json()))).pair_w == BLEND.pair_w


def test_second_look_gate(tmp_path, monkeypatch):
    marker, guard, replay = tmp_path / "second.json", tmp_path / "g.json", tmp_path / "done.json"
    monkeypatch.setattr(PX, "GATE_SECOND", marker)
    monkeypatch.setattr(PX, "GATE_GUARD_V2", guard)
    monkeypatch.setattr(PX, "GATE_DONE", replay)
    with pytest.raises(PermissionError, match="after the oot_week replay"):
        PX.second_look_gate()
    replay.write_text('{"oot_week_scored": true}')
    with pytest.raises(PermissionError, match="frozen guard"):
        PX.second_look_gate()
    guard.write_text("{}")
    PX.second_look_gate()
    marker.write_text("{}")
    with pytest.raises(PermissionError, match="already done"):
        PX.second_look_gate()


def test_filtered_copy_removes_only_the_window(tmp_path):
    t = pd.date_range("2026-10-01 10:00", periods=60, freq="min", tz="UTC")
    src, dst = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pd.DataFrame({"message_time": t, "x": range(60)}).to_parquet(src)
    cut = [(pd.Timestamp("2026-10-01 10:20", tz="UTC"), pd.Timestamp("2026-10-01 10:35", tz="UTC"))]
    Engine._filtered_copy(src, dst, cut)
    got = pd.read_parquet(dst)
    assert len(got) == 45 and got.x.min() == 0 and not got.x.between(20, 34).any()
