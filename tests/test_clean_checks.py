import ast
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa

from rail_risk.clean import checks as ck
from rail_risk.config import load_routes

ROOT = Path(__file__).parent.parent
UTC = "UTC"


def ts(text: str | None):
    return pd.Timestamp(text, tz="UTC") if text else pd.NaT


def test_no_crs_or_route_literal_in_the_cleaning_and_extract_code():
    routes = load_routes(ROOT / "config" / "routes.yaml").routes
    banned = {r.id for r in routes}
    for r in routes:
        banned |= {r.origin, r.destination} | ({r.via_hub} if r.via_hub else set())
    files = list((ROOT / "src/rail_risk/clean").glob("*.py")) + [
        ROOT / "src/rail_risk/ingest/darwin_pport.py",
        ROOT / "src/rail_risk/ingest/pport_extract.py",
        ROOT / "src/rail_risk/ingest/stations.py",
    ]
    offenders = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if v in banned or (re.fullmatch(r"[A-Z]{3}", v) and v != UTC):
                    offenders.append(f"{path.name}: {v!r}")
    assert not offenders, offenders


def sched(rid, **kw):
    base = {
        "rid": rid,
        "ssd": pd.Timestamp("2026-10-14").date(),
        "toc": "XX",
        "is_passenger": True,
        "train_cat": "OO",
        "is_active": True,
        "deleted": False,
        "cancel_reason": pd.NA,
        "message_time": ts("2026-10-13 10:00"),
    }
    return {**base, **kw}


def test_closure_classes_tell_closure_from_missing_data():
    latest = pd.DataFrame(
        [
            sched("1"),  # ran
            sched("2", cancel_reason=100, deleted=True),  # cancelled, with a reason
            sched("3", deleted=True),  # cancelled, no reason
            sched("4"),  # scheduled, nothing came back: missing data
            sched("5", is_passenger=False),
            sched("6", train_cat="BR"),  # replacement bus: never has train actuals
        ]
    )
    points = pd.DataFrame({"rid": ["1", "4"], "cancelled": [False, False]})
    cls = ck.closure_classes(latest, points, actual_rids={"1"}).set_index("rid")["class"]
    assert cls["1"] == "closure_ran_with_actuals"
    assert cls["2"] == "closure_cancellation_with_reason"
    assert cls["3"] == "closure_cancellation_no_reason"
    assert cls["4"] == "closure_missing_data"
    assert cls["5"] == "not_passenger"
    assert cls["6"] == "not_passenger"


def test_cancelled_point_makes_a_schedule_cancelled():
    latest = pd.DataFrame([sched("1")])
    points = pd.DataFrame({"rid": ["1", "1"], "cancelled": [False, True]})
    cls = ck.closure_classes(latest, points, actual_rids=set()).set_index("rid")["class"]
    assert cls["1"] == "closure_cancellation_no_reason"


def test_sequence_failures_count_negative_dwell_and_early_arrival():
    rows = []

    def stop(rid, tpl, wt, arr, dep):
        w = ts(wt)
        if arr:
            rows.append(
                {
                    "rid": rid,
                    "tpl": tpl,
                    "event": "arr",
                    "wt_arr": w,
                    "wt_dep": w,
                    "wt_pass": ts(None),
                    "actual": ts(arr),
                }
            )
        if dep:
            rows.append(
                {
                    "rid": rid,
                    "tpl": tpl,
                    "event": "dep",
                    "wt_arr": w,
                    "wt_dep": w,
                    "wt_pass": ts(None),
                    "actual": ts(dep),
                }
            )

    stop("r", "S1", "2026-10-14 10:00", None, "2026-10-14 10:01")
    stop("r", "S2", "2026-10-14 10:10", "2026-10-14 10:12", "2026-10-14 10:11")  # dep before arr
    stop("r", "S3", "2026-10-14 10:20", "2026-10-14 10:05", None)  # arrives before S2 left
    actuals = pd.DataFrame(rows)
    points = pd.DataFrame(
        {
            "rid": ["r"] * 3,
            "seq": [0, 1, 2],
            "wt_arr": [ts("2026-10-14 10:00"), ts("2026-10-14 10:10"), ts("2026-10-14 09:00")],
            "wt_dep": [pd.NaT] * 3,
            "wt_pass": [pd.NaT] * 3,
        }
    )
    out = ck.sequence_failures(actuals, points)
    assert out["impossible_negative_dwell"]["r"] == 1
    assert out["impossible_arrival_before_prev_departure"]["r"] == 1
    assert out["impossible_schedule_time_decreasing"]["r"] == 1


def test_duplicate_update_rids_counts_exact_repeats_only():
    t = pa.table(
        {
            "rid": ["a", "a", "a", "b"],
            "tpl": ["X"] * 4,
            "est": [1, 1, 2, 1],
            "src_file": ["f1", "f2", "f1", "f1"],
        }
    )
    out = ck.duplicate_update_rids(t)
    assert out.to_dict() == {"a": 1}


def test_schema_flags_bad_rid_and_utc_range():
    df = pd.DataFrame(
        {
            "rid": ["202610147000001", "bad"],
            "ssd": [pd.Timestamp("2026-10-14").date()] * 2,
            "tpl": ["AAAAAAA", "AAAAAAA"],
            "event": ["arr", "arr"],
            "actual": [ts("2026-10-14 10:00"), ts("2026-10-20 10:00")],
            "pt_arr": [pd.NaT] * 2,
            "pt_dep": [pd.NaT] * 2,
            "wt_arr": [pd.NaT] * 2,
            "wt_dep": [pd.NaT] * 2,
            "wt_pass": [pd.NaT] * 2,
            "split": ["train_pool"] * 2,
        }
    )
    bad = ck.schema_failures(df, ck.STOP_ACTUALS)
    assert list(bad) == [False, True]
    assert list(ck.utc_range_failures(df, ["actual"])) == [False, True]
