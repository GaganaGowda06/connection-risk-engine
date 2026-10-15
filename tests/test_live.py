import pandas as pd
import pytest

from rail_risk.config import load_routes
from rail_risk.live import (
    FeatureService,
    LiveStore,
    actual_time_agreement,
    resolve_route,
    source_parity,
)
from world import (
    RID,
    TIPLOCS,
    TODAY,
    events_for,
    history_labels,
    make_store,
    ts,
    ts_message,
    write_raw_hour,
)

ROUTE_DEST = {"bhm-eus": "EUS", "ast-bhm": "BHM", "bhm-man": "MAN"}


def hsp_table(started: bool = True, shift_minutes: int = 0) -> pd.DataFrame:
    """The same service as events_for, in the HSP stop schema."""
    rows = [
        ("BHM", None, "10:00", f"10:{2 + shift_minutes:02d}" if started else None, None),
        ("COV", "10:30", "10:32", None, None),
        ("EUS", "11:30", None, None, None),
    ]
    out = []
    for seq, (loc, arr, dep, act_dep, _) in enumerate(rows):
        out.append(
            {
                "rid": RID, "date_of_service": pd.Timestamp(TODAY), "toc_code": "VT", "seq": seq,
                "location": loc,
                "sched_arr": ts(arr) if arr else pd.NaT, "sched_dep": ts(dep) if dep else pd.NaT,
                "actual_arr": pd.NaT,
                "actual_dep": ts(act_dep) if act_dep else pd.NaT,
                "late_canc_reason": None, "candidate_cancelled": False,
            }
        )  # fmt: skip
    if started:
        out[1]["actual_arr"], out[1]["actual_dep"] = ts("10:33"), ts("10:35")
    return pd.DataFrame(out)


def test_stop_table_has_calling_points_in_order_and_only_real_actuals(tmp_path):
    store = make_store(tmp_path, events_for())
    t = store.stop_table(RID)
    assert t["location"].tolist() == ["BHM", "COVNTRY", "EUS"]  # CRS where mapped
    assert t["seq"].tolist() == [0, 1, 2]
    assert t["toc_code"].iloc[0] == "VT" and not t["candidate_cancelled"].any()
    assert t.loc[t.location == "EUS", "actual_arr"].isna().all()  # a forecast is not an actual
    assert t.loc[t.location == "BHM", "actual_dep"].iloc[0] == ts("10:02")
    assert set(t.columns) >= {"rid", "sched_arr", "actual_dep", "candidate_cancelled"}


def test_tail_of_the_current_hour_is_read_tolerantly_and_deduplicated(tmp_path):
    raw = tmp_path / "raw" / TODAY.isoformat()
    msgs = [
        ts_message(RID, 0, 10, "BHAMNWS", "10:00", "10:02"),
        ts_message(RID, 0, 11, "COVNTRY", "10:32", "10:35"),
        ts_message(RID, 0, 10, "BHAMNWS", "10:00", "10:02"),  # the same Kafka message twice
    ]
    write_raw_hour(raw / "10.jsonl.gz", msgs)
    write_raw_hour(
        raw / "11.jsonl.gz",
        [*msgs, ts_message(RID, 0, 12, "EUSTON", "11:30", "11:29")],
        truncate_last=12,
    )  # file cut mid-write, as when the capture is appending
    store = LiveStore(tmp_path / "interim", tmp_path / "raw", TIPLOCS)
    stats = store.refresh(today=TODAY)
    assert stats.tail_files == 2
    ev = store.events
    assert ev.duplicated(["partition", "offset", "rid", "tiploc", "event_type"]).sum() == 0
    assert set(ev["tiploc"]) == {"BHAMNWS", "COVNTRY"}  # the cut-off last message was dropped
    assert store.has(RID)
    again = store.refresh(today=TODAY)
    assert again.tail_new_lines == 0  # nothing new is reparsed
    assert len(store.events) == len(ev)


def test_cancellation_reading_handles_missing_values():
    assert LiveStore._cancelled({}) == (False, None)
    assert LiveStore._cancelled({"cancel_reason": float("nan"), "n_stops": 5,
                                 "n_cancelled_stops": 0}) == (False, None)  # fmt: skip
    assert LiveStore._cancelled({"cancel_reason": "832", "n_stops": 5,
                                 "n_cancelled_stops": 2}) == (True, "832")  # fmt: skip
    assert LiveStore._cancelled({"cancel_reason": None, "n_stops": 3,
                                 "n_cancelled_stops": 3}) == (True, None)  # fmt: skip


def test_route_resolution_uses_stop_order():
    routes = load_routes("config/routes.yaml").routes
    table = pd.DataFrame({"seq": [0, 1, 2], "location": ["BHM", "COV", "EUS"]})
    assert resolve_route(table, routes) == "bhm-eus"
    assert resolve_route(table, routes, prefer="bhm-man") is None
    backwards = pd.DataFrame({"seq": [0, 1], "location": ["EUS", "BHM"]})
    assert resolve_route(backwards, routes) is None


def test_serving_features_come_from_the_training_code_and_match_across_sources(spark, tmp_path):
    fs = FeatureService.from_labels(spark, ROUTE_DEST, history_labels(spark))
    darwin = make_store(tmp_path, events_for()).stop_table(RID)
    pairs = {RID: (hsp_table(), darwin)}
    out = source_parity(fs, pairs, "bhm-eus", minutes=45, tol=1.0)
    assert out["services"] == 1 and out["agree_share"] == 1.0
    feats = fs.features(darwin, "bhm-eus", 45)
    assert feats["last_stop_lateness_min"] == 3.0  # Coventry departed 10:35 against 10:32
    assert feats["n_stops_passed"] == 2  # origin and Coventry have actuals by T
    assert feats["hist_n_28d"] > 0 and feats["horizon_min"] == 45
    # a feed that differs by 5 minutes is reported as a disagreement
    drift = hsp_table()
    drift.loc[drift.location == "COV", ["actual_arr", "actual_dep"]] = [ts("10:38"), ts("10:40")]
    bad = source_parity(fs, {RID: (drift, darwin)}, "bhm-eus", minutes=45)
    assert bad["agree_share"] == 0.0 and bad["disagreements"][0]["rid"] == RID


def test_actual_time_agreement_counts_matches_and_one_sided_actuals():
    h, d = hsp_table(), hsp_table()
    d.loc[d.location == "BHM", "actual_dep"] = ts("10:05")  # 3 minutes apart
    res = actual_time_agreement({RID: (h, d)})
    assert res["compared_actuals"] == 3
    assert res["identical_share"] == pytest.approx(2 / 3)
    assert res["within_tolerance_share"] == pytest.approx(2 / 3)
    d2 = hsp_table(started=False)
    one_sided = actual_time_agreement({RID: (h, d2)})
    assert one_sided["actual_in_hsp_only"] == 3 and one_sided["compared_actuals"] == 0
