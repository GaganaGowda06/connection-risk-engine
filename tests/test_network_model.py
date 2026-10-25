"""Stage 3: banned identity features, shuffle invariance, paired bootstrap, helpers."""

import json
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F
from rail_risk.network_model import run as network_run
from rail_risk.network_model import train as T
from rail_risk.network_model.evaluate import paired_mae_diff

IDENTITY = ["route", "rid", "toc", "dest_tpl", "from_tpl", "origin_tpl", "hub", "operator"]


def frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in F.FEATURES})
    df["horizon_min"] = rng.choice([5, 15, 60], n)
    df["target_min"] = df["late_now_min"] * 0.8 + rng.normal(size=n)
    df["ssd"] = [date(2026, 9, 1 + i % 5) for i in range(n)]
    df["route"] = rng.choice(["r1", "r2", "r3"], n)
    df["rid"] = [f"id{i}" for i in range(n)]
    df["toc"] = rng.choice(["XC", "VT", "LE"], n)
    df["dest_tpl"] = rng.choice(["AAA", "BBB"], n)
    df["from_tpl"] = rng.choice(["AAA", "BBB"], n)
    return df


def test_the_feature_list_has_no_identity_names():
    F.assert_clean_feature_names(F.FEATURES)
    for bad in IDENTITY + ["route_id", "toc_code", "CRS", "tiploc_x"]:
        with pytest.raises(ValueError, match="banned"):
            F.assert_clean_feature_names([*F.FEATURES, bad])


def test_the_matrix_carries_only_features_whatever_the_frame_holds():
    df = frame()
    x = D.matrix(df)
    assert list(x.columns) == F.FEATURES
    assert not set(IDENTITY) & set(x.columns)
    with pytest.raises(ValueError, match="banned"):
        D.matrix(df, [*F.FEATURES, "toc"])


def test_a_trained_booster_sees_no_identity_and_ignores_shuffled_labels():
    df = frame()
    x = D.matrix(df)
    params = {"objective": "quantile", "alpha": 0.5, "verbose": -1, "num_threads": 1, "seed": 1}
    booster = lgb.train(params, lgb.Dataset(x, df["target_min"]), 30)
    names = booster.feature_name()
    assert names == F.FEATURES
    assert not any(b in n.lower() for n in names for b in F.BANNED_SUBSTRINGS)
    before = booster.predict(D.matrix(df))
    shuffled = df.copy()
    rng = np.random.default_rng(3)
    for c in ("route", "rid", "toc", "dest_tpl", "from_tpl"):
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
    assert np.array_equal(before, booster.predict(D.matrix(shuffled)))


MODELS = sorted(Path("data/interim/stage3/models").glob("*/meta.joblib"))


@pytest.mark.skipif(not MODELS, reason="no trained network model on disk")
def test_the_saved_model_files_carry_only_clean_feature_names():
    for meta in MODELS:
        model = T.Booster3.load(meta.parent)
        for b in [*model.quantiles.values(), model.classifier]:
            assert not any(s in n.lower() for n in b.feature_name() for s in F.BANNED_SUBSTRINGS), (
                meta
            )


def test_paired_bootstrap_detects_a_better_model_and_a_tie():
    rng = np.random.default_rng(1)
    ssd = np.repeat(np.arange(10), 50)
    base = rng.normal(0, 3, 500)
    better = paired_mae_diff(base * 0.5, base, ssd, 500, 1)
    assert better["hi"] < 0 and better["n_dates"] == 10
    tie = paired_mae_diff(base, base, ssd, 500, 1)
    assert tie["diff"] == 0 and tie["lo"] == 0 and tie["hi"] == 0
    one_day = paired_mae_diff(base, base * 2, np.zeros(500), 100, 1)
    assert np.isnan(one_day["lo"])


def test_slack_is_scheduled_minus_fastest_and_nan_for_unseen_pairs():
    feats = pd.DataFrame(
        {
            "from_tpl": ["A", "B"], "from_event": "dep", "dest_tpl": ["Z", "Z"],
            "sched_min_remaining": [30.0, 20.0], "late_now_min": [12.0, 3.0],
        }
    )  # fmt: skip
    fastest = pd.DataFrame(
        {"from_tpl": ["A"], "from_event": "dep", "dest_tpl": ["Z"], "fastest_min": [24.0]}
    )
    out = F.add_slack(feats, fastest)
    assert out["slack_min"].iloc[0] == 6.0 and out["late_minus_slack_min"].iloc[0] == 6.0
    assert np.isnan(out["slack_min"].iloc[1])


def test_fastest_table_is_the_fifth_percentile_and_needs_enough_observations(tmp_path):
    d = date(2026, 9, 1)
    base = pd.Timestamp("2026-09-01 10:00", tz="UTC")
    rows = []
    for i in range(30):
        run = 20 + (i % 10)
        rows.append((f"r{i}", "A", "dep", base))
        rows.append((f"r{i}", "Z", "arr", base + pd.Timedelta(minutes=run)))
    rows.append(("lone", "B", "dep", base))
    rows.append(("lone", "Z", "arr", base + pd.Timedelta(minutes=9)))
    df = pd.DataFrame(rows, columns=["rid", "tpl", "event", "actual"]).assign(split="train_pool")
    part = tmp_path / "stop_actuals" / f"service_date={d}"
    part.mkdir(parents=True)
    df.to_parquet(part / "x.parquet")
    pairs = pd.DataFrame({"from_tpl": ["A", "B"], "from_event": "dep", "dest_tpl": "Z"})
    out = F.fastest_table(tmp_path, [d], pairs)
    assert list(out["from_tpl"]) == ["A"]  # B has one observation, under the minimum
    assert 20.0 <= out["fastest_min"].iloc[0] <= 21.0


def test_route_folds_partition_the_routes():
    routes = [f"r{i}" for i in range(18)]
    folds = T.route_folds(routes, 4)
    assert sorted(set().union(*folds)) == sorted(routes)
    assert sum(len(f) for f in folds) == len(routes)


def test_target_transforms_round_trip():
    df = pd.DataFrame({"target_min": [5.0, -1.0], "late_now_min": [3.0, np.nan]})
    base = np.array([2.0, 0.5])
    for option in ("residual", "raw", "recovery"):
        y = T.target_for_fit(option, df, base)
        assert np.allclose(T.to_lateness(option, df, base, y), df["target_min"].to_numpy())


def test_the_test_block_cannot_be_scored_twice_or_without_a_passed_gate(tmp_path, monkeypatch):
    marker = tmp_path / "test_scored.json"
    marker.write_text(json.dumps({"run_id": "x"}))
    monkeypatch.setattr(network_run, "TEST_MARKER", marker)
    with pytest.raises(RuntimeError, match="already scored"):
        network_run.evaluate_command("x")
