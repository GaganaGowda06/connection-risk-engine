import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "stage5_label_audit", Path(__file__).parents[1] / "scripts" / "stage5_label_audit.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_unknown_needs_no_arrival_no_cancel_and_no_deactivation():
    t = pd.Timestamp("2026-10-12 10:00")
    df = pd.DataFrame(
        {
            "hub": ["BHM"] * 5,
            "in_rid": ["a", "b", "c", "d", "e"],
            "miss": [1.0, 1.0, 1.0, 0.0, 1.0],
            "out_cancel": [0] * 5,
            "act_arr_in": [pd.NaT, pd.NaT, pd.NaT, t, t],
            "in_cancel_final": [False, True, False, False, False],
        }
    )
    g = audit.classify(
        df, deact={"b", "c", "d", "e"}
    )  # only a has no arrival, no cancel, no deactivation
    row = g.loc["BHM"]
    assert row["to_unknown"] == 1 and row["missing_arrival"] == 3
    assert row["missing_and_cancelled"] == 1 and row["variant_ignoring_deactivation"] == 2
