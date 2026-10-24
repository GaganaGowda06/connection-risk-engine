from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rail_risk.config import load_routes
from rail_risk.stress_eval import (
    JOURNEY_LABELS,
    _fmt,
    clean,
    flag_short_history,
    journey_bucket,
    load_stress_config,
    prior_fetched_days,
)

ROOT = Path(__file__).resolve().parents[1]
HUB_ROUTES = {
    "bhm-man": "MAN",
    "bhm-lds": "LDS",
    "bhm-bri": "BRI",
    "bhm-liv": "LIV",
    "bhm-glc": "GLC",
}


def rows(route: str, first: date, days: int, per_day: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"route_id": route, "service_date": first + timedelta(days=d)}
            for d in range(days)
            for _ in range(per_day)
        ]
    )


def test_hub_routes_in_config_run_from_birmingham():
    routes = {r.id: r for r in load_routes(ROOT / "config" / "routes.yaml").routes}
    for route_id, crs in HUB_ROUTES.items():
        assert routes[route_id].origin == "BHM"
        assert routes[route_id].destination == crs
        assert routes[route_id].via_hub is None


def test_stress_config_routes_are_configured_and_disjoint():
    cfg = load_stress_config(ROOT / "config" / "stress.yaml")
    known = {r.id for r in load_routes(ROOT / "config" / "routes.yaml").routes}
    assert set(cfg.own_routes) | set(cfg.new_routes) <= known
    assert set(cfg.new_routes) == set(HUB_ROUTES)
    assert not set(cfg.own_routes) & set(cfg.new_routes)
    assert cfg.min_prior_days == 7


def test_prior_days_count_distinct_fetched_days_before_each_date():
    df = rows("r", date(2026, 10, 1), 10)
    prior = prior_fetched_days(df["route_id"].to_numpy(), df["service_date"].to_numpy())
    by_day = dict(zip(df["service_date"], prior, strict=True))
    assert by_day[date(2026, 10, 1)] == 0
    assert by_day[date(2026, 10, 2)] == 1
    assert by_day[date(2026, 10, 8)] == 7
    assert by_day[date(2026, 10, 10)] == 9


def test_flag_marks_fewer_than_seven_prior_days_and_not_exactly_seven():
    df = flag_short_history(rows("r", date(2026, 10, 1), 10), min_days=7)
    flagged = {d for d, f in zip(df["service_date"], df["short_history"], strict=True) if f}
    assert flagged == {date(2026, 10, d) for d in range(1, 8)}
    assert df.loc[df["service_date"] == date(2026, 10, 8), "short_history"].eq(False).all()


def test_history_counts_only_the_28_day_window_and_only_fetched_days():
    # a fetched day 30 days back is outside the window; a day with no rows never counts
    old = rows("r", date(2026, 9, 1), 1)
    recent = pd.concat([rows("r", date(2026, 10, 3), 2), rows("r", date(2026, 10, 7), 1)])
    df = pd.concat([old, recent], ignore_index=True)
    prior = prior_fetched_days(df["route_id"].to_numpy(), df["service_date"].to_numpy())
    by_day = dict(zip(df["service_date"], prior, strict=True))
    assert by_day[date(2026, 10, 7)] == 2  # the 3rd and 4th, not the 1st of September


def test_history_is_counted_per_route():
    df = pd.concat([rows("a", date(2026, 10, 1), 5), rows("b", date(2026, 10, 4), 2)])
    flagged = flag_short_history(df.reset_index(drop=True), min_days=7)
    b = flagged[flagged["route_id"] == "b"]
    assert b["prior_days"].tolist() == [0, 0, 1, 1]


def test_flag_never_uses_the_row_date_or_later_days():
    df = rows("r", date(2026, 10, 1), 3)
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    out = flag_short_history(shuffled, min_days=1)
    assert out.loc[out["service_date"] == date(2026, 10, 1), "prior_days"].eq(0).all()
    assert out.loc[out["service_date"] == date(2026, 10, 3), "prior_days"].eq(2).all()


def test_journey_buckets():
    got = journey_bucket(np.array([10.0, 30.0, 74.0, 90.0, 150.0, 240.0, np.nan]))
    assert list(got) == [
        JOURNEY_LABELS[0], JOURNEY_LABELS[1], JOURNEY_LABELS[1], JOURNEY_LABELS[2],
        JOURNEY_LABELS[3], JOURNEY_LABELS[4], "unknown",
    ]  # fmt: skip


def test_clean_makes_results_json_safe():
    out = clean({"a": np.float64("nan"), "b": np.int64(3), "c": (date(2026, 10, 1),)})
    assert out == {"a": None, "b": 3, "c": ["2026-10-01"]}


def test_fmt_handles_missing_and_interval():
    assert _fmt(None) == "n/a"
    assert _fmt({"value": None}) == "n/a"
    assert _fmt({"value": 1.0, "ci_low": 0.5, "ci_high": 1.5}) == "1.00 [0.50, 1.50]"


def test_bad_window_rejected(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text(
        "own: {routes: [a], start: 2026-10-22, end: 2026-10-16}\n"
        "new: {routes: [b], start: 2026-10-01, end: 2026-10-22}\n"
        "horizons: [15]\nmin_prior_days: 7\nleakage_rows: 50\n"
    )
    with pytest.raises(ValueError, match="ends before"):
        load_stress_config(bad)


def test_fetched_dates_before_the_scored_rows_count_as_history():
    # an own route scored only for the last days, with the earlier days fetched but not scored
    df = rows("r", date(2026, 10, 16), 3)
    earlier = [date(2026, 10, 1) + timedelta(days=d) for d in range(15)]
    out = flag_short_history(df, min_days=7, fetched={"r": [*earlier, *df["service_date"]]})
    assert out["prior_days"].min() >= 15
    assert not out["short_history"].any()
    alone = flag_short_history(df, min_days=7)
    assert alone["short_history"].all()
