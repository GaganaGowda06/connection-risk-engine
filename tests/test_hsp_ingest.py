import json
from datetime import date
from pathlib import Path

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
    rids_from_metrics,
)
from rail_risk.ingest.hsp_parse import parse_details, to_frame, validate

FIX = Path(__file__).parent / "fixtures"
ROUTE = Route(id="bhm-eus", origin="BHM", destination="EUS")


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
    assert first["sched_dep"] == pd.Timestamp("2026-10-01 07:06").to_pydatetime()
    assert last["location"] == "EUS"
    assert last["actual_arr"] == pd.Timestamp("2026-10-01 09:26").to_pydatetime()
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
    assert rows[1]["sched_arr"] == pd.Timestamp("2026-10-01 23:58").to_pydatetime()
    assert rows[1]["actual_arr"] == pd.Timestamp("2026-10-02 00:03").to_pydatetime()
    assert rows[2]["sched_arr"] == pd.Timestamp("2026-10-02 00:30").to_pydatetime()
    assert rows[2]["actual_arr"] == pd.Timestamp("2026-10-02 00:41").to_pydatetime()


def test_early_actual_before_midnight_stays_on_previous_day():
    rows = parse_details(details([loc("BHM", ptd="0002", atd="2359"), loc("EUS", pta="0100")]))
    assert rows[0]["actual_dep"] == pd.Timestamp("2026-09-30 23:59").to_pydatetime()


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
