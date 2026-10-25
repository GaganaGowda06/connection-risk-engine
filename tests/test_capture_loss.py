import gzip
import json
from datetime import date

from rail_risk.capture_loss import (
    archive_file,
    archive_files_for,
    backfill,
    scan_archive,
    scan_capture,
    summarise,
)

DAY = date(2026, 10, 20)  # BST: UTC+1


def cap_line(offset: int, ts: str, rid: str) -> bytes:
    rec = {
        "partition": 0,
        "offset": offset,
        "kafka_ts": 0,
        "payload": {"ts": ts, "uR": {"TS": {"rid": rid}}},
    }
    return json.dumps(rec, separators=(",", ":")).encode()


def arch_doc(ts: str, rid: str) -> bytes:
    return f'<?xml version="1.0"?><Pport ts="{ts}"><uR><TS rid="{rid}"/></uR></Pport>'.encode()


def put(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(b"\n".join(lines) + b"\n")


def test_archive_files_follow_uk_local_hour():
    assert archive_file(DAY, 10) == (DAY, 11)  # 10:xx UTC is 11:xx BST
    assert archive_file(date(2026, 10, 25), 0) == (date(2026, 10, 25), 1)
    assert archive_files_for(DAY)[23] == (date(2026, 10, 21), 0)


def test_messages_are_bucketed_by_utc_hour_of_their_own_timestamp(tmp_path):
    arch = tmp_path / "a.gz"
    put(arch, [arch_doc("2026-10-20T11:05:00.123+01:00", "202610200000001")])
    s = scan_archive([arch], DAY)
    assert s.msgs[10] == 1 and 202610200000001 == int(next(iter(s.rids[10])))
    other = tmp_path / "o.gz"
    put(other, [arch_doc("2026-10-21T00:30:00+01:00", "202610200000002")])  # 23:30 UTC on the 20th
    assert scan_archive([other], DAY).msgs[23] == 1


def test_replayed_offsets_count_as_duplicates_and_never_as_captured(tmp_path):
    root = tmp_path / "cap"
    lines = [cap_line(i, "2026-10-20T11:05:00+01:00", f"20261020000000{i}") for i in range(4)]
    put(root / "2026-10-20" / "10.jsonl.gz", lines)
    put(
        root / "2026-10-20" / "11.jsonl.gz",
        lines[2:] + [cap_line(9, "2026-10-20T12:00:00+01:00", "202610200000009")],
    )
    s = scan_capture(root, DAY)
    assert s.duplicates == 2 and s.msgs[10] == 4 and s.msgs[11] == 1


def test_loss_gap_hours_and_backfill_tag(tmp_path):
    arch_docs = [arch_doc("2026-10-20T11:05:00+01:00", f"20261020000000{i}") for i in range(10)]
    arch_path = tmp_path / "11.pport.gz"
    put(arch_path, arch_docs)
    root = tmp_path / "cap"
    put(
        root / "2026-10-20" / "10.jsonl.gz",
        [cap_line(i, "2026-10-20T11:05:00+01:00", f"20261020000000{i}") for i in range(7)],
    )
    report = summarise(DAY, scan_archive([arch_path], DAY), scan_capture(root, DAY))
    assert report["loss_pct"] == 30.0
    assert report["gap_hours"] == [10] and report["rid_loss_pct"] == 30.0
    written = backfill(report["gap_hours"], DAY, {10: arch_path}, tmp_path / "bf")
    dest = tmp_path / "bf" / "2026-10-20" / "10.pport.gz"
    assert written[0]["source"] == "archive_backfill" and dest.exists()
    tag = json.loads((dest.parent / "10.pport.gz.source.json").read_text())
    assert tag["source"] == "archive_backfill"
    # backfill is not under the capture root, so a rescan still reports the same loss
    assert (
        summarise(DAY, scan_archive([arch_path], DAY), scan_capture(root, DAY))["loss_pct"] == 30.0
    )
