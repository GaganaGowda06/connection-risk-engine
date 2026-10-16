"""Parsing closed Darwin hours from the API, the lost-schedule skew it fixes, and the
stale-parse check."""

import gzip
import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from rail_risk.ingest.darwin import DarwinManifest, parse_directory
from rail_risk.live import LiveStore
from rail_risk.status import STALE_PARSE_HOURS, darwin_status, parse_lag
from test_connection_api import INBOUND, client, connect, local, make

TIPLOCS = {"BHAMNWS": "BHM", "EUSTON": "EUS"}
RID = "202610150000777"
DAY = date(2026, 10, 15)


def line(offset: int, body: dict) -> bytes:
    payload = {
        "ts": "2026-10-15T09:00:00+01:00",
        "version": "18.0",
        "uR": {"updateOrigin": "CIS", **body},
    }
    return json.dumps(
        {"partition": 0, "offset": offset, "kafka_ts": 0, "payload": payload}
    ).encode()


def write_hour(raw: Path, hour: str, lines: list[bytes]) -> None:
    path = raw / DAY.isoformat() / f"{hour}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(b"\n".join(lines) + b"\n"))


SCHEDULE = {
    "schedule": {
        "rid": RID, "uid": "B00777", "ssd": "2026-10-15", "toc": "VT",
        "OR": {"tpl": "BHAMNWS", "wtd": "10:00", "ptd": "10:00"},
        "DT": {"tpl": "EUSTON", "wta": "11:20", "pta": "11:20"},
    }
}  # fmt: skip
MOVEMENT = {
    "TS": {
        "rid": RID, "uid": "B00777", "ssd": "2026-10-15",
        "Location": {"tpl": "BHAMNWS", "ptd": "10:00", "dep": {"at": "10:02", "src": "TD"}},
    }
}  # fmt: skip


def world(tmp_path: Path) -> tuple[Path, Path]:
    """The schedule arrives in hour 08 UTC, the first movement of the service in hour 09."""
    raw, interim = tmp_path / "raw", tmp_path / "interim"
    write_hour(raw, "08", [line(1, SCHEDULE)])
    write_hour(raw, "09", [line(2, MOVEMENT)])
    return raw, interim


def test_tail_reading_drops_a_schedule_that_arrived_in_an_earlier_hour(tmp_path):
    raw, interim = world(tmp_path)
    store = LiveStore(interim, raw, TIPLOCS)
    store.refresh(today=DAY)
    assert store.has(RID)  # the movement is there
    assert RID not in store.schedules  # but the earlier schedule was dropped: operator unknown


def test_reload_after_parsing_recovers_the_schedule(tmp_path):
    raw, interim = world(tmp_path)
    store = LiveStore(interim, raw, TIPLOCS)
    store.refresh(today=DAY)
    manifest = DarwinManifest(interim / "darwin_manifest.sqlite")
    now = datetime(2026, 10, 15, 10, 30, tzinfo=UTC)  # hours 08 and 09 are closed
    assert len(parse_directory(raw, interim, manifest, now=now)) == 2
    manifest.close()
    store.reload_parsed(today=DAY)
    assert store.schedules[RID]["toc"] == "VT"
    before = len(store.events)
    store.reload_parsed(today=DAY)  # repeating changes nothing
    assert len(store.events) == before and store.schedules[RID]["toc"] == "VT"


def test_service_parses_new_closed_hours_once_and_updates_the_lag(tmp_path, features, bundle_pair):
    service = make(tmp_path, features, bundle_pair, warm=False)
    raw, interim = world(tmp_path)  # the same directories the service's store reads
    service.store.raw, service.store.interim = raw, interim
    service.parse_enabled = True
    service.now = lambda: datetime(2026, 10, 15, 10, 30, tzinfo=UTC)
    assert service.parse_closed_hours() == 2
    assert service.parse_state.newest_parsed == "2026-10-15/09" and not service.parse_state.stale
    assert service.parse_closed_hours() == 0  # nothing new: it only lists files
    write_hour(raw, "10", [line(3, MOVEMENT)])  # hour 10 is current, so it is skipped
    assert service.parse_closed_hours() == 0
    service.now = lambda: datetime(2026, 10, 15, 11, 5, tzinfo=UTC)
    assert service.parse_closed_hours() == 1  # hour 10 has closed


def test_parsing_is_off_when_not_enabled_but_the_lag_is_still_checked(
    tmp_path, features, bundle_pair
):
    service = make(tmp_path, features, bundle_pair, warm=False)
    raw, interim = world(tmp_path)
    service.store.raw, service.store.interim = raw, interim
    assert service.parse_closed_hours() == 0 and not (interim / "darwin").exists()
    assert service.parse_state.stale and service.parse_state.newest_parsed is None


def seed_manifest(interim: Path, keys: list[str]) -> Path:
    interim.mkdir(parents=True, exist_ok=True)
    DarwinManifest(interim / "darwin_manifest.sqlite").close()
    db = sqlite3.connect(interim / "darwin_manifest.sqlite")
    for k in keys:
        db.execute("INSERT INTO files (key, size) VALUES (?, 1)", (k,))
    db.commit()
    db.close()
    return interim / "darwin_manifest.sqlite"


def raw_hours(raw: Path, hours: list[str]) -> None:
    for h in hours:
        write_hour(raw, h, [b"{}"])


@pytest.mark.parametrize(
    ("raw_hours_", "parsed", "stale", "lag"),
    [
        (["10"], ["2026-10-15/09"], False, 1.0),  # healthy: the current hour is never parsed
        (["10", "11"], ["2026-10-15/09"], False, 2.0),  # exactly the limit is still fine
        (["12"], ["2026-10-15/09"], True, 3.0),
        (["09", "10"], ["2026-10-14/23"], True, 11.0),  # across midnight
        (["10"], [], True, None),  # nothing parsed at all
    ],
)
def test_parse_lag(tmp_path, raw_hours_, parsed, stale, lag):
    raw = tmp_path / "raw"
    raw_hours(raw, raw_hours_)
    manifest = seed_manifest(tmp_path / "interim", parsed)
    got = parse_lag(raw, manifest)
    assert got.stale is stale and got.lag_hours == lag and STALE_PARSE_HOURS == 2.0


def test_no_raw_files_is_not_stale_and_a_missing_manifest_is(tmp_path):
    assert parse_lag(tmp_path / "raw", tmp_path / "m.sqlite").stale is False
    raw_hours(tmp_path / "raw", ["10"])
    assert parse_lag(tmp_path / "raw", tmp_path / "m.sqlite").stale is True


def test_status_prints_the_parse_lag_and_flags_it(tmp_path):
    raw = tmp_path / "raw"
    raw_hours(raw, ["12"])
    seed_manifest(tmp_path / "interim", ["2026-10-15/09"])
    text = "\n".join(darwin_status(raw, tmp_path / "interim"))
    assert "parse lag: newest raw 2026-10-15/12, newest parsed 2026-10-15/09, 3 h behind" in text
    assert "STALE" in text
    seed_manifest(tmp_path / "interim", ["2026-10-15/11"])
    assert "STALE" not in "\n".join(darwin_status(raw, tmp_path / "interim"))


def test_stale_parse_shows_in_health_and_in_every_data_quality_block(
    tmp_path, features, bundle_pair
):
    service = make(tmp_path, features, bundle_pair)
    raw = tmp_path / "raw"
    raw_hours(raw, ["12"])
    manifest = seed_manifest(tmp_path / "interim", ["2026-10-15/08"])
    service.store.raw = raw
    assert manifest.exists() and service.parse_closed_hours() == 0  # lag only, parsing is off
    with client(service) as c:
        health = c.get("/health").json()
        assert health["stale_parse"] is True and health["parse_lag_hours"] == 4.0
        assert health["status"] == "degraded"
        body = connect(c, outbound_departure=local("10:47")).json()
        dq = body["data_quality"]
        assert dq["stale_parse"] is True and any("waiting to be parsed" in n for n in dq["notes"])
        risk = c.post("/risk", json={"rid": INBOUND}).json()
        assert risk["data_quality"]["stale_parse"] is True


def test_fresh_parse_leaves_health_ok_and_stale_parse_false(tmp_path, features, bundle_pair):
    service = make(tmp_path, features, bundle_pair)
    with client(service) as c:
        health = c.get("/health").json()
        assert health["stale_parse"] is False and health["parse_lag_hours"] is None
        dq = connect(c, outbound_departure=local("10:47")).json()["data_quality"]
        assert dq["stale_parse"] is False
        assert not any("waiting to be parsed" in n for n in dq["notes"])
