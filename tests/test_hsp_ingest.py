import json
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pytest
from tenacity import wait_none

from rail_risk.config import Route
from rail_risk.ingest.hsp import (
    Counters,
    HspClient,
    HspHttpError,
    Manifest,
    backfill,
    metrics_body,
    rebuild_interim,
    rids_from_metrics,
)
from rail_risk.ingest.hsp_parse import parse_details, to_frame, validate

FIX = Path(__file__).parent / "fixtures"
ROUTE = Route(id="bhm-eus", origin="BHM", destination="EUS")


def utc(local: str) -> datetime:
    """UTC instant for a London wall clock time, computed independently of the parser."""
    naive = datetime.fromisoformat(local)
    return naive.replace(tzinfo=ZoneInfo("Europe/London")).astimezone(UTC)


def load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def details(locations: list[dict], rid="202610018980803", day="2026-10-01") -> dict:
    return {
        "serviceAttributesDetails": {
            "date_of_service": day,
            "toc_code": "VT",
            "rid": rid,
            "locations": locations,
        }
    }


def loc(code, pta="", ptd="", ata="", atd="", reason=""):
    return {
        "location": code,
        "gbtt_pta": pta,
        "gbtt_ptd": ptd,
        "actual_ta": ata,
        "actual_td": atd,
        "late_canc_reason": reason,
    }


# parsing


def test_parse_real_service():
    rows = parse_details(load("hsp_details_ontime.json"))
    assert len(rows) == 15
    first, last = rows[0], rows[-1]
    assert first["location"] == "BHM" and first["seq"] == 0
    assert first["sched_arr"] is None
    assert first["sched_dep"] == utc("2026-10-01 07:06")
    assert last["location"] == "EUS"
    assert last["actual_arr"] == utc("2026-10-01 09:26")
    assert last["candidate_cancelled"] is False


def test_reason_codes_kept():
    rows = parse_details(load("hsp_details_reason.json"))
    assert rows[-1]["late_canc_reason"] == "520"
    assert rows[0]["late_canc_reason"] is None


def test_midnight_rollover_scheduled_and_actual():
    payload = details(
        [
            loc("BHM", ptd="2350", atd="2352"),
            loc("RUG", pta="2358", ptd="2359", ata="0003", atd="0004"),
            loc("EUS", pta="0030", ata="0041"),
        ]
    )
    rows = parse_details(payload)
    assert rows[1]["sched_arr"] == utc("2026-10-01 23:58")
    assert rows[1]["actual_arr"] == utc("2026-10-02 00:03")
    assert rows[2]["sched_arr"] == utc("2026-10-02 00:30")
    assert rows[2]["actual_arr"] == utc("2026-10-02 00:41")


def test_early_actual_before_midnight_stays_on_previous_day():
    rows = parse_details(details([loc("BHM", ptd="0002", atd="2359"), loc("EUS", pta="0100")]))
    assert rows[0]["actual_dep"] == utc("2026-09-30 23:59")


def test_ambiguous_time_first_occurrence_by_default():
    # 2026-10-25 01:30 happens twice; with nothing earlier in the service, take BST (00:30Z)
    rows = parse_details(details([loc("BHM", ptd="0130", atd="0130"), loc("EUS", pta="0300")],
                                 day="2026-10-25"))  # fmt: skip
    assert rows[0]["sched_dep"] == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)
    assert rows[0]["actual_dep"] == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)


def test_ambiguous_time_second_occurrence_when_sequence_demands_it():
    # 01:55 BST (00:55Z) then 01:05 on the same wall clock day can only be 01:05 GMT (01:05Z)
    stops = [loc("BHM", ptd="0155"), loc("RUG", pta="0105", ptd="0106"), loc("EUS", pta="0230")]
    rows = parse_details(details(stops, day="2026-10-25"))
    assert rows[0]["sched_dep"] == datetime(2026, 10, 25, 0, 55, tzinfo=UTC)
    assert rows[1]["sched_arr"] == datetime(2026, 10, 25, 1, 5, tzinfo=UTC)
    assert rows[2]["sched_arr"] == datetime(2026, 10, 25, 2, 30, tzinfo=UTC)  # 02:30 GMT


def test_nonexistent_time_shifts_forward_by_the_gap():
    # 2026-03-29 01:30 does not exist; it is read with the pre-change offset: 01:30Z (02:30 BST)
    rows = parse_details(
        details([loc("BHM", ptd="0130"), loc("EUS", pta="0300")], day="2026-03-29")
    )
    assert rows[0]["sched_dep"] == datetime(2026, 3, 29, 1, 30, tzinfo=UTC)
    assert rows[1]["sched_arr"] == datetime(2026, 3, 29, 2, 0, tzinfo=UTC)  # 03:00 BST


def test_parsed_timestamps_are_utc_aware():
    df = to_frame(parse_details(load("hsp_details_ontime.json")))
    assert str(df["sched_dep"].dtype) == "datetime64[ns, UTC]"


def test_candidate_cancelled_when_destination_has_no_actuals():
    rows = parse_details(details([loc("BHM", ptd="0700", atd="0701"), loc("EUS", pta="0830")]))
    assert all(r["candidate_cancelled"] for r in rows)


def test_details_without_locations_give_no_rows():
    assert parse_details(details([])) == []


# validation and quarantine


def frame_for(locations, rid="202610018980803"):
    return to_frame(parse_details(details(locations, rid=rid)))


def test_valid_real_service_passes():
    good, bad = validate(to_frame(parse_details(load("hsp_details_ontime.json"))))
    assert len(good) == 15 and bad.empty


def test_actual_more_than_six_hours_from_scheduled_is_quarantined():
    df = frame_for([loc("BHM", ptd="0700", atd="1400"), loc("EUS", pta="0830", ata="0840")])
    good, bad = validate(df)
    assert good.empty
    assert len(bad) == 2  # the whole service is quarantined
    assert "within_6h" in bad["quarantine_reason"].iloc[0]


def test_bad_rid_and_location_quarantined_other_service_kept():
    ok = frame_for([loc("BHM", ptd="0700", atd="0701"), loc("EUS", pta="0830", ata="0835")])
    bad_rid = frame_for(
        [loc("BHM", ptd="0700", atd="0701"), loc("e1s", pta="0830", ata="0835")], rid="12345"
    )
    good, bad = validate(pd.concat([ok, bad_rid], ignore_index=True))
    assert set(good["rid"]) == {"202610018980803"}
    assert set(bad["rid"]) == {"12345"}
    assert bad["quarantine_reason"].str.len().min() > 0


def test_duplicate_stop_quarantined():
    df = frame_for([loc("BHM", ptd="0700", atd="0701"), loc("EUS", pta="0830", ata="0835")])
    good, bad = validate(pd.concat([df, df], ignore_index=True))
    assert good.empty and len(bad) == 4


# client and backfill


def make_client(handler, **kw) -> HspClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return HspClient(
        "test-key", http=http, min_gap_s=0, retry_wait=wait_none(), sleep=lambda s: None, **kw
    )


def test_retries_on_5xx_and_429_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503 if len(calls) == 1 else 429 if len(calls) == 2 else 200, json={})

    client = make_client(handler)
    assert client.post("serviceMetrics", {}) == {}
    assert len(calls) == 3
    assert client.counters.retries == 2


def test_gives_up_after_three_attempts():
    client = make_client(lambda r: httpx.Response(500, json={}))
    with pytest.raises(Exception, match="HTTP 500"):
        client.post("serviceMetrics", {})
    assert client.counters.metrics_calls == 3


def test_4xx_is_not_retried_and_hides_key():
    client = make_client(lambda r: httpx.Response(401, text="Invalid ApiKey"))
    with pytest.raises(HspHttpError) as err:
        client.post("serviceMetrics", {})
    assert client.counters.metrics_calls == 1
    assert "test-key" not in str(err.value)


def test_min_gap_between_calls():
    sleeps, now = [], [0.0]
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    client = HspClient(
        "k",
        http=http,
        min_gap_s=1.0,
        sleep=sleeps.append,
        clock=lambda: now[0],
        retry_wait=wait_none(),
    )
    client.post("serviceMetrics", {})
    now[0] = 0.25
    client.post("serviceMetrics", {})
    assert sleeps == [pytest.approx(0.75)]


def test_metrics_body_days_and_window():
    assert metrics_body(ROUTE, date(2026, 10, 3), 6) == {
        "from_loc": "BHM", "to_loc": "EUS", "from_time": "0600", "to_time": "0700",
        "from_date": "2026-10-03", "to_date": "2026-10-03", "days": "SATURDAY",
    }  # fmt: skip
    assert metrics_body(ROUTE, date(2026, 10, 2), 21)["days"] == "WEEKDAY"


def test_rids_from_metrics():
    assert len(rids_from_metrics(load("hsp_metrics.json"))) == 5


def test_backfill_then_rerun_makes_no_calls(tmp_path):
    metrics = load("hsp_metrics.json")
    detail = load("hsp_details_ontime.json")
    seen = []

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        seen.append(endpoint)
        if endpoint == "serviceMetrics":
            return httpx.Response(200, json=metrics)
        rid = json.loads(request.content)["rid"]
        payload = json.loads(json.dumps(detail))
        payload["serviceAttributesDetails"]["rid"] = rid
        return httpx.Response(200, json=payload)

    raw, interim = tmp_path / "raw", tmp_path / "interim"
    day = date(2026, 10, 1)

    def run() -> Counters:
        client = make_client(handler)
        manifest = Manifest(interim / "m.sqlite")
        try:
            return backfill(
                ROUTE, day, day, range(7, 9), client, manifest, raw_root=raw, interim_root=interim
            )
        finally:
            manifest.close()

    first = run()
    # hour 7 fetches 5 rids; hour 8 returns the same rids, which are skipped as already fetched
    assert (first.metrics_calls, first.detail_calls) == (2, 5)
    assert first.rows == 5 * 15 and first.quarantined_rows == 0
    assert (raw / "metrics" / "2026-10-01" / "bhm-eus_07.json").exists()
    assert len(list((raw / "details" / "2026-10-01").glob("*.json"))) == 5
    assert (interim / "hsp_stops" / "date=2026-10-01" / "bhm-eus_07.parquet").exists()

    seen.clear()
    second = run()
    assert seen == []
    assert second.windows_skipped == 2 and second.windows_done == 0


def test_crash_mid_window_resumes_from_raw_without_refetching(tmp_path):
    metrics = load("hsp_metrics.json")
    detail = load("hsp_details_ontime.json")
    raw, interim = tmp_path / "raw", tmp_path / "interim"
    fail_after = [2]

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "serviceMetrics":
            return httpx.Response(200, json=metrics)
        if fail_after[0] == 0:
            return httpx.Response(401, text="boom")
        fail_after[0] -= 1
        rid = json.loads(request.content)["rid"]
        payload = json.loads(json.dumps(detail))
        payload["serviceAttributesDetails"]["rid"] = rid
        return httpx.Response(200, json=payload)

    day = date(2026, 10, 1)
    manifest = Manifest(interim / "m.sqlite")
    with pytest.raises(HspHttpError):
        backfill(ROUTE, day, day, range(7, 8), make_client(handler), manifest,
                 raw_root=raw, interim_root=interim)  # fmt: skip
    assert not manifest.window_done("bhm-eus", day, 7)
    assert len(list((raw / "details" / "2026-10-01").glob("*.json"))) == 2

    fail_after[0] = 99
    client = make_client(handler)
    c = backfill(ROUTE, day, day, range(7, 8), client, manifest, raw_root=raw, interim_root=interim)
    assert c.detail_calls == 3  # only the 3 missing details
    assert manifest.window_done("bhm-eus", day, 7)
    manifest.close()


def test_one_rid_serves_two_routes_without_refetching(tmp_path):
    metrics = load("hsp_metrics.json")
    detail = load("hsp_details_ontime.json")
    other = Route(id="ast-eus", origin="AST", destination="EUS", via_hub="BHM")
    seen = []

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        seen.append(endpoint)
        if endpoint == "serviceMetrics":
            return httpx.Response(200, json=metrics)
        payload = json.loads(json.dumps(detail))
        payload["serviceAttributesDetails"]["rid"] = json.loads(request.content)["rid"]
        return httpx.Response(200, json=payload)

    raw, interim = tmp_path / "raw", tmp_path / "interim"
    day = date(2026, 10, 1)
    manifest = Manifest(interim / "m.sqlite")
    for route in (ROUTE, other):
        backfill(route, day, day, range(7, 8), make_client(handler), manifest,
                 raw_root=raw, interim_root=interim)  # fmt: skip
    assert seen.count("serviceDetails") == 5  # fetched once, for the first route
    assert seen.count("serviceMetrics") == 2
    members = manifest.query("SELECT route, COUNT(*) FROM rid_routes GROUP BY route ORDER BY route")
    assert members == [("ast-eus", 5), ("bhm-eus", 5)]
    assert manifest.query("SELECT COUNT(*) FROM rids") == [(5,)]
    manifest.close()


def test_rebuild_from_raw_makes_no_calls_and_matches_rows(tmp_path):
    metrics = load("hsp_metrics.json")
    detail = load("hsp_details_ontime.json")

    def handler(request):
        if request.url.path.endswith("serviceMetrics"):
            return httpx.Response(200, json=metrics)
        payload = json.loads(json.dumps(detail))
        payload["serviceAttributesDetails"]["rid"] = json.loads(request.content)["rid"]
        return httpx.Response(200, json=payload)

    raw, interim = tmp_path / "raw", tmp_path / "interim"
    day = date(2026, 10, 1)
    manifest = Manifest(interim / "m.sqlite")
    c = backfill(ROUTE, day, day, range(7, 8), make_client(handler), manifest,
                 raw_root=raw, interim_root=interim)  # fmt: skip
    for parquet in interim.rglob("*.parquet"):
        parquet.unlink()
    rows, bad = rebuild_interim(manifest, raw, interim)
    assert (rows, bad) == (c.rows, 0)
    assert len(list((interim / "hsp_stops").rglob("*.parquet"))) == 1
    manifest.close()


def test_manifest_migrates_route_membership_from_old_rids_table(tmp_path):
    import sqlite3

    db = tmp_path / "m.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE rids (rid TEXT PRIMARY KEY, route TEXT, day TEXT, hour INTEGER)")
    con.execute("INSERT INTO rids VALUES ('202610018980803','bhm-eus','2026-10-01',7)")
    con.commit()
    con.close()
    m = Manifest(db)
    assert m.query("SELECT rid, route FROM rid_routes") == [("202610018980803", "bhm-eus")]
    m.close()


def test_newest_first_orders_days_descending(tmp_path):
    days = []

    def handler(request):
        days.append(json.loads(request.content)["from_date"])
        return httpx.Response(200, json={"Services": []})

    manifest = Manifest(tmp_path / "m.sqlite")
    backfill(ROUTE, date(2026, 10, 1), date(2026, 10, 3), range(7, 8), make_client(handler),
             manifest, raw_root=tmp_path / "raw", interim_root=tmp_path / "i",
             newest_first=True)  # fmt: skip
    assert days == ["2026-10-03", "2026-10-02", "2026-10-01"]
    manifest.close()
