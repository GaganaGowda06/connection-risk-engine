import json

import numpy as np
import pandas as pd
import pytest

from rail_risk.drift import (
    ACT,
    WATCH,
    drift_report,
    psi_categorical,
    psi_numeric,
    status,
    write_report,
)


def frame(n: int, seed: int, shift: float = 0.0, missing: float = 0.0, cat_p: float = 0.5):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n) + shift
    x = np.where(rng.random(n) < missing, np.nan, x)
    return pd.DataFrame(
        {
            "x": x,
            "const": 3.0,
            "toc_code": np.where(rng.random(n) < cat_p, "VT", "LM"),
        }
    )


def test_same_distribution_has_low_psi_and_a_shift_has_high_psi():
    train, same, shifted = frame(20_000, 1), frame(5_000, 2), frame(5_000, 3, shift=1.0)
    low = psi_numeric(train["x"], same["x"])
    high = psi_numeric(train["x"], shifted["x"])
    assert low["psi"] < WATCH and low["bins"] == 10
    assert high["psi"] > ACT
    assert status(low["psi"]) == "ok" and status(high["psi"]) == "act"
    assert status(0.15) == "watch" and status(ACT) == "act"


def test_psi_grows_with_the_size_of_the_shift():
    train = frame(20_000, 1)
    values = [psi_numeric(train["x"], frame(5_000, 4, shift=s)["x"])["psi"] for s in (0, 0.3, 0.8)]
    assert values == sorted(values) and values[2] > 3 * values[1] > 0


def test_missing_values_are_a_bin_of_their_own():
    train, live = frame(20_000, 1), frame(5_000, 2, missing=0.4)
    r = psi_numeric(train["x"], live["x"])
    assert r["psi"] > ACT and r["live_missing"] == pytest.approx(0.4, abs=0.03)
    assert r["train_missing"] == 0.0


def test_values_outside_the_training_range_are_counted_not_dropped():
    train = pd.Series(np.linspace(0, 1, 1000))
    live = pd.Series(np.full(500, 50.0))
    assert psi_numeric(train, live)["psi"] > ACT


def test_constant_feature_is_stable_and_unseen_categories_are_pooled():
    train, live = frame(2_000, 1), frame(1_000, 2)
    assert psi_numeric(train["const"], live["const"])["psi"] == pytest.approx(0.0)
    unseen = pd.Series(["XX"] * 500)
    assert psi_categorical(train["toc_code"], unseen)["psi"] > ACT
    flipped = frame(5_000, 5, cat_p=0.9)["toc_code"]
    assert psi_categorical(train["toc_code"], flipped)["psi"] > ACT
    assert psi_categorical(train["toc_code"], live["toc_code"])["psi"] < WATCH


def test_report_ranks_features_flags_skips_and_handles_thin_live_data(tmp_path):
    train = frame(20_000, 1)
    live = frame(3_000, 2, shift=1.0)
    rep = drift_report(
        train, live, ["x", "const", "absent"], ["toc_code"],
        window={"live_from": "2026-10-09", "live_to": "2026-10-15", "minutes": 60},
    )  # fmt: skip
    names = [r["feature"] for r in rep["features"]]
    assert names[0] == "x" and rep["features"][0]["status"] == "act"
    assert "absent" in rep["skipped"]
    assert rep["summary"]["act"] >= 1 and rep["summary"]["ok"] >= 1
    write_report(rep, tmp_path)
    data = json.loads((tmp_path / "drift.json").read_text())
    assert data["thresholds"] == {"watch": WATCH, "act": ACT}
    md = (tmp_path / "drift.md").read_text()
    assert "| x |" in md and "—" not in md and "–" not in md
    thin = drift_report(train, live.head(5), ["x"], [])
    assert thin["features"] == [] and "fewer than" in thin["note"]
    write_report(thin, tmp_path)
    assert "fewer than" in (tmp_path / "drift.md").read_text()
