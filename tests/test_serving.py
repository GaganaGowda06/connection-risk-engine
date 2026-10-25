"""Stage 6: the oot_week gate, the state stores, the freshness signals and the API boundary."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rail_risk.ingest import pport_extract as P
from rail_risk.serving import engine as E
from rail_risk.serving import state as S


def test_oot_week_stays_closed_without_a_threshold_file(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "GATE_THRESHOLD", tmp_path / "guard_threshold.json")
    monkeypatch.setattr(P, "GATE_DONE", tmp_path / "done.json")
    with pytest.raises(PermissionError, match="closed"):
        P.reading_split("oot_week", "oot_replay").__enter__()
    (tmp_path / "guard_threshold.json").write_text("{}")
    with P.reading_split("oot_week", "oot_replay"):
        assert P._ACTIVE_SPLIT == "oot_week"
    assert P._ACTIVE_SPLIT == "train_pool"
    (tmp_path / "done.json").write_text("{}")
    with pytest.raises(PermissionError, match="already"):
        P.reading_split("oot_week", "oot_replay").__enter__()


def test_oot_week_is_refused_for_any_other_purpose(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "GATE_THRESHOLD", tmp_path / "t.json")
    (tmp_path / "t.json").write_text("{}")
    for purpose in ("tuning", "february_stress", "scoring"):
        with pytest.raises(ValueError):
            P.reading_split("oot_week", purpose).__enter__()


def test_archive_hour_labels_are_local_hours():
    assert E.label_to_utc("2026-10-12_01.parquet") == pd.Timestamp("2026-10-12 00:00", tz="UTC")
    assert E.label_to_utc("2026-01-12_01.parquet") == pd.Timestamp("2026-01-12 01:00", tz="UTC")


FRAME = pd.DataFrame(
    {"rid": ["a", "b"], "horizon_min": [15, 60], "x": [1.5, np.nan], "stale": [True, False],
     "snap_time": pd.to_datetime(["2026-10-01 12:00", "2026-10-01 13:00"], utc=True)}
)  # fmt: skip


@pytest.mark.parametrize("kind", ["memory", "duckdb", "sqlite"])
def test_stores_commit_read_and_resume(kind, tmp_path):
    st = S.open_store(kind, tmp_path / kind)
    st.commit_hour("h1", {"arrival": FRAME}, {"k": 1})
    st.commit_hour("h2", {"arrival": FRAME.assign(rid=["c", "d"])}, {"k": 2})
    assert st.hours_done() == ["h1", "h2"] and st.load_blob() == {"k": 2}
    assert len(st.risk("a")) == 1 and st.risk("zzz").empty
    assert len(st.all("arrival")) == 4
    st.close()
    again = S.open_store(kind, tmp_path / kind)
    assert again.hours_done() == ["h1", "h2"] and again.load_blob() == {"k": 2}
    again.close()


@pytest.mark.parametrize("kind", ["duckdb", "sqlite"])
def test_an_hour_is_all_or_nothing(kind, tmp_path):
    st = S.open_store(kind, tmp_path / kind)
    st.commit_hour("h1", {"arrival": FRAME}, {"k": 1})
    with pytest.raises(Exception):  # noqa: B017  (the same hour twice breaks the hours key)
        st.commit_hour("h1", {"arrival": FRAME}, {"k": 2})
    assert len(st.all("arrival")) == 2 and st.load_blob() == {"k": 1}


def test_freshness_signals_see_a_gap():
    minutes = np.array(list(range(0, 100)) + list(range(160, 200)), dtype="int64")
    t = pd.Series(pd.to_datetime([100 * 60 + 5 * 60, 130 * 60 * 1, 170 * 60], unit="s", utc=True))
    # 105 min after epoch (age 6 since last message at minute 99), 130 min (age 31), 170 min (age 0)
    t = pd.Series(pd.to_datetime([105 * 60, 130 * 60, 170 * 60], unit="s", utc=True))
    age = E.Engine._age(minutes, t)
    assert age.tolist() == [6.0, 31.0, 0.0]
    gap = E.Engine._recent_gap(minutes, t, 120)
    assert gap.tolist() == [6.0, 31.0, 61.0]  # the 61 minute hole is still inside the window at 170


def test_thresholds_default_to_never_stale(tmp_path):
    th = E.Thresholds.load(tmp_path / "missing.json")
    assert th.feed_min > 1e6 and th.hub_min > 1e6


def test_api_validates_input_and_reports_no_data(tmp_path, monkeypatch):
    from rail_risk.serving import app as A

    monkeypatch.setattr(A, "STATE_DIR", tmp_path)
    c = TestClient(A.app)
    assert c.post("/risk", json={"rid": "not-a-rid"}).status_code == 422
    assert c.post("/connection", json={"in_rid": "1", "out_rid": "2"}).status_code == 422
    assert c.get("/inbound", params={"hub": "XYZ"}).status_code == 422
    h = c.get("/health").json()
    assert h["stale"] is True and h["status"] == "degraded" and h["data_as_of"] is None
    assert c.post("/risk", json={"rid": "202610010000001"}).status_code == 404
