import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

from rail_risk.capture import GapDetector, HourlyWriter, hour_path, strip_message


def ms(*args) -> int:
    return int(datetime(*args, tzinfo=UTC).timestamp() * 1000)


def wrapper(payload: dict) -> bytes:
    return json.dumps(
        {"destination": {"name": "q"}, "messageID": "ID:1", "bytes": json.dumps(payload)}
    ).encode()


def test_strip_keeps_payload_and_kafka_position():
    payload = {"ts": "2026-10-15T11:21:21+01:00", "uR": {"TS": {"rid": "1"}}}
    line = strip_message(wrapper(payload), partition=3, offset=42, kafka_ts_ms=1000)
    rec = json.loads(line)
    assert rec == {"partition": 3, "offset": 42, "kafka_ts": 1000, "payload": payload}
    assert b"\n" not in line
    assert b"messageID" not in line


def test_strip_keeps_raw_when_not_a_wrapper():
    for value in (b"not json", b'{"no_bytes": 1}', b'{"bytes": "also not json"}', None):
        rec = json.loads(strip_message(value, 0, 1, 5))
        assert "payload" not in rec
        assert rec["raw"] == (value or b"").decode()


def test_hour_path_uses_utc_hour_of_kafka_timestamp():
    root = Path("data/raw/darwin")
    assert hour_path(root, ms(2026, 10, 15, 9, 59, 59)) == root / "2026-10-15" / "09.jsonl.gz"
    assert hour_path(root, ms(2026, 10, 15, 23, 0, 0)) == root / "2026-10-15" / "23.jsonl.gz"
    assert hour_path(root, ms(2026, 10, 16, 0, 0, 0)) == root / "2026-10-16" / "00.jsonl.gz"


def test_writer_rolls_on_hour_and_never_goes_back(tmp_path):
    w = HourlyWriter(tmp_path)
    assert w.write(ms(2026, 10, 15, 10, 5), b'{"a":1}') is False
    assert w.write(ms(2026, 10, 15, 10, 59), b'{"a":2}') is False
    assert w.write(ms(2026, 10, 15, 11, 0), b'{"a":3}') is True
    # a late message from the earlier hour goes to the current file, with no second roll
    assert w.write(ms(2026, 10, 15, 10, 58), b'{"a":4}') is False
    w.close()
    ten = gzip.open(tmp_path / "2026-10-15" / "10.jsonl.gz", "rt").read().splitlines()
    eleven = gzip.open(tmp_path / "2026-10-15" / "11.jsonl.gz", "rt").read().splitlines()
    assert ten == ['{"a":1}', '{"a":2}']
    assert eleven == ['{"a":3}', '{"a":4}']


def test_writer_appends_on_restart(tmp_path):
    for n in (1, 2):
        w = HourlyWriter(tmp_path)
        w.write(ms(2026, 10, 15, 10, 0), json.dumps({"n": n}).encode())
        w.close()
    path = tmp_path / "2026-10-15" / "10.jsonl.gz"
    assert gzip.open(path, "rt").read().splitlines() == ['{"n": 1}', '{"n": 2}']


def test_gap_detector():
    g = GapDetector(threshold_s=300)
    assert g.check(ms(2026, 10, 15, 10, 0)) is None  # first message
    assert g.check(ms(2026, 10, 15, 10, 4, 59)) is None  # under threshold
    assert g.check(ms(2026, 10, 15, 10, 10, 0)) == 301.0  # jump of 5 min 1 s
    assert g.check(ms(2026, 10, 15, 10, 10, 1)) is None
