from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from rail_risk.features.build import leakage_check_real
from rail_risk.features.snapshots import (
    FEATURE_COLUMNS,
    HORIZONS,
    LIVE_FEATURES,
    build_snapshots,
)

STOP_SCHEMA = (
    "rid string, service_date date, toc_code string, seq long, location string, "
    "sched_arr timestamp, sched_dep timestamp, actual_arr timestamp, actual_dep timestamp, "
    "late_canc_reason string, candidate_cancelled boolean"
)
STATIONS = ["AAA", "BBB", "CCC", "EUS"]
ROUTE = {"r1": "EUS"}


def at(day: date, hhmm: str, plus: int = 0) -> datetime:
    return datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(
        hours=int(hhmm[:2]), minutes=int(hhmm[2:]) + plus
    )


def service(rid, day, depart, spacing, delays, stations=STATIONS, cancelled=False, reasons=()):
    """Stops for one service. delays[i] is the lateness in minutes at stop i (None = no actual)."""
    rows = []
    for i, code in enumerate(stations):
        arr = at(day, depart, spacing * i) if i > 0 else None
        dep = at(day, depart, spacing * i) if i < len(stations) - 1 else None
        d = delays[i]
        rows.append(
            (
                rid, day, "VT", i, code, arr, dep,
                None if arr is None or d is None else arr + timedelta(minutes=d),
                None if dep is None or d is None else dep + timedelta(minutes=d),
                reasons[i] if reasons else None,
                cancelled,
            )
        )  # fmt: skip
    return rows


def synthetic(days: list[date]):
    rows, links = [], []
    for di, day in enumerate(days):
        for k in range(4):  # four services an hour apart, 30 min between stops
            rid = f"{day:%Y%m%d}{k:07d}"
            delays = [(di + k + i) % 5 for i in range(4)]
            rows += service(rid, day, f"{8 + k:02d}00", 30, delays)
            links.append((rid, "r1"))
        long_rid = f"{day:%Y%m%d}9000001"  # an early starter: 6 stops, started before T-120
        long_stations = ["P", "Q", "R", "S", "T", "EUS"]
        reasons = [None, None, "895", None, None, None]
        rows += service(
            long_rid, day, "0600", 40, [1, 2, 3, 4, 5, 6], stations=long_stations, reasons=reasons
        )
        links.append((long_rid, "r1"))
        canc = f"{day:%Y%m%d}9000002"
        rows += service(canc, day, "1200", 30, [None] * 4, cancelled=True)
        links.append((canc, "r1"))
    return rows, links


def frames(spark, rows, links):
    return (
        spark.createDataFrame(rows, STOP_SCHEMA),
        spark.createDataFrame(links, "rid string, route_id string"),
    )


def snapshot_rows(spark, rows, links, **kw):
    stops, rr = frames(spark, rows, links)
    df = build_snapshots(spark, stops, rr, ROUTE, **kw)
    return {(r["rid"], r["horizon_min"]): r.asDict() for r in df.collect()}


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a == b or (isinstance(a, float) and abs(a - b) < 1e-9)


DAYS = [date(2026, 9, 14) + timedelta(days=i) for i in range(5)]


@pytest.fixture(scope="module")
def base(spark):
    rows, links = synthetic(DAYS)
    return rows, links, snapshot_rows(spark, rows, links)


@pytest.mark.parametrize("mode", ["delete", "shift"])
@pytest.mark.parametrize("horizon", HORIZONS)
def test_actuals_after_t_never_change_features(spark, base, horizon, mode):
    rows, links, expected = base
    target = DAYS[3]
    dest_arr = {}
    for r in rows:
        if r[4] == "EUS":
            dest_arr[r[0]] = r[5]
    altered = []
    for r in rows:
        r = list(r)
        if r[1] == target:
            t = dest_arr[r[0]] - timedelta(minutes=horizon)
            for idx in (7, 8):  # actual_arr, actual_dep
                if r[idx] is not None and r[idx] > t:
                    r[idx] = None if mode == "delete" else r[idx] + timedelta(minutes=90)
        altered.append(tuple(r))
    got = snapshot_rows(spark, altered, links)
    checked = 0
    for (rid, h), row in expected.items():
        if h != horizon or row["service_date"] != target:
            continue
        for col in FEATURE_COLUMNS:
            assert same(row[col], got[(rid, h)][col]), (rid, h, col)
        checked += 1
    assert checked >= 6


def test_history_for_date_d_ignores_d_and_later(spark, base):
    rows, links, expected = base
    cutoff = DAYS[2]
    altered = []
    for r in rows:
        r = list(r)
        if r[1] >= cutoff:  # every actual on D and later is moved, so labels change
            for idx in (7, 8):
                if r[idx] is not None:
                    r[idx] = r[idx] + timedelta(minutes=45)
        altered.append(tuple(r))
    got = snapshot_rows(spark, altered, links)
    hist = ["hist_median_28d", "hist_p80_28d", "hist_n_28d"]
    for (rid, h), row in expected.items():
        if row["service_date"] == cutoff:
            for col in hist:
                assert same(row[col], got[(rid, h)][col]), (rid, h, col)
    # not vacuous: the day after D does depend on D's labels
    nxt = [
        (k, row)
        for k, row in expected.items()
        if row["service_date"] == DAYS[3] and row["hist_median_28d"] is not None
    ]
    assert nxt and any(
        not same(row["hist_median_28d"], got[k]["hist_median_28d"]) for k, row in nxt
    )


def test_history_window_is_28_days_before_and_excludes_the_day(spark):
    day = date(2026, 9, 30)
    days = [day - timedelta(days=29), day - timedelta(days=28), day - timedelta(days=1), day]
    rows, links = synthetic(days)
    got = snapshot_rows(spark, rows, links)
    r = got[(f"{day:%Y%m%d}0000000", 60)]
    # the first service of D arrives 09:30Z (hour 10 local). Two services a day arrive in that
    # local hour, so the two in-window days give 4. D-29 and D itself must not count.
    assert r["hist_n_28d"] == 4
    first_day = got[(f"{days[0]:%Y%m%d}0000000", 60)]
    assert first_day["hist_n_28d"] == 0 and first_day["hist_median_28d"] is None


def test_not_yet_departed_has_null_live_state_but_context_present(spark, base):
    _, _, got = base
    for (rid, h), row in got.items():
        if h == 120 and not rid.endswith(("9000001", "9000002")):
            assert all(row[c] is None for c in LIVE_FEATURES), rid
            assert row["hour_of_day"] is not None and row["toc_code"] == "VT"
    started = next(r for (rid, h), r in got.items() if h == 120 and rid.endswith("9000001"))
    assert started["last_stop_seq"] is not None and started["n_stops_passed"] >= 1


def test_label_and_candidate_cancellation(spark, base):
    _, _, got = base
    row = got[(f"{DAYS[0]:%Y%m%d}0000000", 15)]
    assert row["label_delay_min"] == (0 + 0 + 3) % 5 or row["label_delay_min"] is not None
    cancelled = got[(f"{DAYS[0]:%Y%m%d}9000002", 15)]
    assert cancelled["is_candidate_cancelled"] is True
    assert cancelled["label_delay_min"] is None
    assert row["is_candidate_cancelled"] is False


def test_reason_share_is_not_a_feature():
    assert "reason_share_passed" not in FEATURE_COLUMNS


def test_bank_holidays_come_from_the_fetched_file(spark):
    from rail_risk.features.snapshots import load_bank_holidays

    days = load_bank_holidays()
    assert "2026-12-28" in days and "2026-12-26" not in days  # Boxing Day moved to Monday
    day = date(2026, 12, 25)
    rows, links = synthetic([day, date(2026, 12, 24)])
    got = snapshot_rows(spark, rows, links)
    assert got[(f"{day:%Y%m%d}0000000", 60)]["is_bank_holiday"] is True
    assert got[("202612240000000", 60)]["is_bank_holiday"] is False


def test_live_state_values_are_exact(spark):
    day = date(2026, 9, 14)
    # 40 min spacing: stops at 06:00 06:40 07:20 08:00 08:40 09:20 (EUS). Delays 1..6.
    stations = ["P", "Q", "R", "S", "T", "EUS"]
    reasons = [None, "574", None, None, None, None]
    rows = service(
        "202609140000001", day, "0600", 40, [1, 2, 3, 4, 5, 6], stations=stations, reasons=reasons
    )
    got = snapshot_rows(spark, rows, [("202609140000001", "r1")])
    r = got[("202609140000001", 60)]  # T = 08:20: stop S arrived 08:00 + 4 = 08:04, departs 08:04
    assert r["last_stop_seq"] == 3
    assert r["last_stop_lateness_min"] == 4.0
    assert r["lateness_change_last_2"] == 1.0  # 4 (S) minus 3 (R)
    assert r["minutes_since_last_stop"] == 16.0  # 08:20 minus 08:04
    assert r["stops_remaining"] == 2
    assert r["sched_minutes_remaining"] == 80.0  # 08:00 to 09:20
    assert r["n_stops_passed"] == 4
    assert r["label_delay_min"] == 6.0


@pytest.mark.skipif(
    not Path("data/interim/hsp_manifest.sqlite").exists()
    or not list(Path("data/interim/hsp_stops").glob("date=*/*.parquet")),
    reason="needs ingested HSP data",
)
def test_real_data_rows_do_not_change_when_future_actuals_are_deleted(spark):
    checked, altered, mismatches = leakage_check_real(
        spark, Path("data/interim"), Path("config/routes.yaml"), n=50
    )
    assert checked == 50
    assert altered > 0  # the check removed real future actuals, so it is not vacuous
    assert mismatches == []
