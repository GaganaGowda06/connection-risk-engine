import numpy as np
import pandas as pd
import pytest

from rail_risk.baselines.target_check import TargetCheckError, assert_targets_ok, check_targets


def frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "rid": [f"r{i}" for i in range(n)],
            "route": ["a", "b"] * (n // 2),
            "horizon_min": 15,
            "target_late_s": rng.gamma(1.5, 120.0, n) - 60.0,
        }
    )


def test_a_clean_frame_passes_and_blocks_are_reported():
    report = assert_targets_ok(None, {"train": frame(), "test": frame(seed=1)})
    assert report["failures"] == [] and report["rows"] == {"train": 400, "test": 400}


@pytest.mark.parametrize(
    ("mutate", "text"),
    [
        (
            lambda d: d.__setitem__(
                "target_late_s", d["target_late_s"].where(d.index > 0, 86460.0)
            ),
            "day-sized",
        ),
        (
            lambda d: d.__setitem__(
                "target_late_s", d["target_late_s"].where(d.index > 0, -86460.0)
            ),
            "day-sized",
        ),
        (
            lambda d: d.__setitem__(
                "target_late_s", d["target_late_s"].where(d.index > 0, -15000.0)
            ),
            "impossible",
        ),
        (lambda d: d.__setitem__("rid", d["rid"].where(d.index > 0, "r1")), "duplicate"),
    ],
)
def test_the_defects_that_once_slipped_through_now_fail_loudly(mutate, text):
    df = frame()
    mutate(df)
    if text == "duplicate":
        df.loc[0, "route"] = df.loc[1, "route"]
    with pytest.raises(TargetCheckError, match=text):
        assert_targets_ok(df)


def test_a_systematic_offset_on_one_route_fails_and_a_small_early_share_only_warns():
    df = frame(1000)
    mask = (df["route"] == "a") & (df.index % 5 == 0)
    df.loc[mask, "target_late_s"] = -3600.0  # 20 percent of route a an hour early
    assert any("under -30" in f for f in check_targets(df)["failures"])
    df = frame(1000)
    mask = (df["route"] == "a") & (df.index % 40 == 0)
    df.loc[mask, "target_late_s"] = -3600.0  # 5 percent of route a: a warning only
    report = check_targets(df)
    assert report["failures"] == [] and any("under -30" in w for w in report["warnings"])
