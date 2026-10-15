import gzip
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from rail_risk.ingest.darwin import (
    DarwinManifest,
    ReadStats,
    iter_lines,
    normalise,
    parse_directory,
    parse_file,
)
from rail_risk.status import darwin_status

FIX = Path(__file__).parent / "fixtures" / "darwin_lines.jsonl"


def fixture_lines() -> list[bytes]:
    return FIX.read_bytes().splitlines()


def line(offset: int, uR: dict, ts="2026-10-15T11:00:00+01:00", partition=0) -> bytes:  # noqa: N803
    payload = {"ts": ts, "version": "18.0", "uR": {"updateOrigin": "CIS", **uR}}
    rec = {"partition": partition, "offset": offset, "kafka_ts": 0, "payload": payload}
    return json.dumps(rec).encode()


def write_gz(path: Path, lines: list[bytes], members: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    per = max(1, len(lines) // members)
    with path.open("wb") as fh:
        for i in range(0, len(lines), per):
            fh.write(gzip.compress(b"\n".join(lines[i : i + per]) + b"\n"))


def tables(interim: Path, name: str) -> pd.DataFrame:
    files = list((interim / "darwin" / name).glob("*/*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run(tmp_path: Path, lines: list[bytes], key="2026-10-15T10"):
    raw = tmp_path / "raw" / "2026-10-15" / "10.jsonl.gz"
    write_gz(raw, lines)
    interim = tmp_path / "interim"
    return parse_file(raw, key, interim), interim


# reading


def test_both_line_formats_normalise():
    lines = fixture_lines()
    old = normalise(lines[0])  # first trial format: JMS wrapper
    new = normalise(lines[1])
    assert old[3]["uR"] and old[2] is None
    assert new[3]["uR"] and isinstance(new[2], int)


def test_truncated_gzip_keeps_complete_lines_and_counts_loss(tmp_path):
    good = [b'{"n":%d,"pad":"%s"}' % (i, b"x" * 200) for i in range(2000)]
    blob = gzip.compress(b"\n".join(good) + b"\n")
    path = tmp_path / "t.jsonl.gz"
    path.write_bytes(blob[: len(blob) // 2])  # a hard kill mid-write
    stats = ReadStats()
    got = list(iter_lines(path, stats))
    assert 0 < len(got) < len(good)
    assert all(json.loads(g)["n"] == i for i, g in enumerate(got))  # complete lines only
    assert stats.truncated is True


def test_partial_final_line_is_counted_as_lost(tmp_path):
    path = tmp_path / "t.jsonl.gz"
    with gzip.open(path, "wb") as fh:
        fh.write(b'{"a":1}\n{"a":2}\n{"a":3')
    stats = ReadStats()
    assert list(iter_lines(path, stats)) == [b'{"a":1}', b'{"a":2}']
    assert stats.bytes_lost == len(b'{"a":3')
    assert stats.truncated is True


def test_multi_member_gzip_reads_through(tmp_path):
    lines = [b'{"a":%d}' % i for i in range(10)]
    path = tmp_path / "m.jsonl.gz"
    write_gz(path, lines, members=5)
    stats = ReadStats()
    assert list(iter_lines(path, stats)) == lines
    assert stats.truncated is False and stats.bytes_lost == 0


# parsing the real fixture lines


def test_fixture_lines_parse_into_all_tables(tmp_path):
    result, interim = run(tmp_path, fixture_lines())
    assert result.stats.unreadable == 0 and result.stats.duplicates == 0
    assert (
        result.rows["darwin_schedules"] == 2
    )  # the cancelled one and one riding in an association line
    assert result.rows["darwin_events"] > 0
    assert result.rows["darwin_associations"] >= 1
    assert sum(result.quarantined.values()) == 0
    events = tables(interim, "darwin_events")
    assert str(events["time"].dtype) == "datetime64[ns, UTC]"
    assert set(events["event_type"]) <= {"arr", "dep", "pass"}


def test_partial_cancellation_in_schedule(tmp_path):
    _, interim = run(tmp_path, fixture_lines())
    schedules = tables(interim, "darwin_schedules")
    sched = schedules[schedules["cancel_reason"].notna()].iloc[0]
    assert sched["cancel_reason"].isdigit()
    assert 0 < sched["n_cancelled_stops"] < sched["n_stops"]
    stops = list(sched["stops"])
    assert [s["seq"] for s in stops] == list(range(len(stops)))
    assert any(s["can"] for s in stops) and not all(s["can"] for s in stops)
    times = [s["wta"] or s["wtp"] or s["wtd"] for s in stops]
    assert times == sorted(times)  # running order restored from grouped JSON


def test_late_reason_both_shapes(tmp_path):
    _, interim = run(tmp_path, fixture_lines())
    events = tables(interim, "darwin_events")
    coded = events[events["late_reason"].notna()]
    assert coded["late_reason"].str.fullmatch(r"\d+").all()
    assert coded["late_reason_tpl"].notna().any()  # the object form carried a tiploc
    assert coded["late_reason_tpl"].isna().any()  # the bare form did not


def test_cancelled_association(tmp_path):
    _, interim = run(tmp_path, fixture_lines())
    assoc = tables(interim, "darwin_associations")
    assert assoc["is_cancelled"].any()
    assert assoc["category"].isin(["JJ", "VV", "LK", "NP"]).all()


# duplicates and quarantine


def test_duplicate_offsets_removed(tmp_path):
    lines = fixture_lines()
    once, _ = run(tmp_path / "a", lines)
    twice, _ = run(tmp_path / "b", [*lines, *lines])
    assert twice.stats.duplicates == len(lines)
    assert twice.rows == once.rows


def test_bad_rows_are_quarantined_not_dropped(tmp_path):
    where = {"tpl": "BHAMNWS", "wtd": "10:00"}
    bad_ts = ts_message([{**where, "dep": {"et": "10:01"}}])
    bad_ts["TS"]["rid"] = "123"
    bad_time = ts_message([{**where, "dep": {"et": "25:99"}}])
    result, interim = run(tmp_path, [line(1, bad_ts), line(2, bad_time)])
    assert result.rows["darwin_events"] == 0
    assert result.quarantined["darwin_events"] == 2
    quarantine = pd.read_parquet(next((interim / "darwin" / "darwin_events_quarantine").glob("*")))
    assert set(quarantine["offset"]) == {1, 2}
    assert quarantine["quarantine_reason"].str.len().min() > 0
    assert any("rid" in r for r in result.reasons)


# time handling


def ts_message(locations: list[dict], ssd="2026-10-15") -> dict:
    return {"TS": {"rid": f"{ssd.replace('-', '')}0000001", "uid": "A00001", "ssd": ssd,
                   "Location": locations}}  # fmt: skip


def events_for(tmp_path, uR) -> pd.DataFrame:  # noqa: N803
    _, interim = run(tmp_path, [line(1, uR)])
    return tables(interim, "darwin_events").sort_values("offset", kind="stable")


def test_midnight_rollover(tmp_path):
    # planned 23:58 BST, forecast 00:03 the next morning
    ev = events_for(
        tmp_path,
        ts_message([{"tpl": "RUGBY", "wtd": "23:58", "ptd": "23:58", "dep": {"et": "00:03"}}]),
    )
    row = ev.iloc[0]
    assert row["planned_time"] == pd.Timestamp("2026-10-15 22:58", tz="UTC")
    assert row["time"] == pd.Timestamp("2026-10-15 23:03", tz="UTC")  # 00:03 BST on the 16th


def test_planned_chain_rolls_past_midnight(tmp_path):
    ev = events_for(
        tmp_path,
        ts_message(
            [
                {"tpl": "AAAA", "wtd": "23:50", "ptd": "23:50", "dep": {"et": "23:51"}},
                {"tpl": "BBBB", "wta": "00:10", "pta": "00:10", "arr": {"et": "00:12"}},
            ]
        ),
    )
    b = ev[ev["tiploc"] == "BBBB"].iloc[0]
    assert b["planned_time"] == pd.Timestamp("2026-10-15 23:10", tz="UTC")  # 00:10 BST on the 16th
    assert b["time"] == pd.Timestamp("2026-10-15 23:12", tz="UTC")


def test_dst_ambiguous_hour(tmp_path):
    # 2026-10-25 clocks go back at 02:00 BST. 01:25 is read as BST (00:25Z); the forecast 01:40
    # is the occurrence nearest the plan (00:40Z). A later planned 01:10 would run backwards,
    # so it is the second occurrence (01:10Z).
    ev = events_for(
        tmp_path,
        ts_message(
            [
                {"tpl": "AAAA", "wta": "01:25", "pta": "01:25", "arr": {"et": "01:40"}},
                {"tpl": "BBBB", "wta": "01:10", "pta": "01:10", "arr": {"et": "01:12"}},
            ],
            ssd="2026-10-25",
        ),
    )
    a = ev[ev["tiploc"] == "AAAA"].iloc[0]
    b = ev[ev["tiploc"] == "BBBB"].iloc[0]
    assert a["planned_time"] == pd.Timestamp("2026-10-25 00:25", tz="UTC")
    assert a["time"] == pd.Timestamp("2026-10-25 00:40", tz="UTC")
    assert b["planned_time"] == pd.Timestamp("2026-10-25 01:10", tz="UTC")
    assert b["time"] == pd.Timestamp("2026-10-25 01:12", tz="UTC")


def test_schedule_order_restored_when_json_groups_ip_and_pp(tmp_path):
    sched = {
        "schedule": {
            "rid": "202610150000009", "uid": "B00009", "ssd": "2026-10-15", "toc": "VT",
            "OR": {"tpl": "ORIG", "wtd": "23:40", "ptd": "23:40"},
            "IP": [{"tpl": "IP1", "wta": "23:50", "wtd": "23:51", "pta": "23:50", "ptd": "23:51"},
                   {"tpl": "IP2", "wta": "00:20", "wtd": "00:21", "pta": "00:20", "ptd": "00:21"}],
            "PP": [{"tpl": "PP1", "wtp": "00:05"}],
            "DT": {"tpl": "DEST", "wta": "00:40", "pta": "00:40"},
        }
    }  # fmt: skip
    _, interim = run(tmp_path, [line(1, sched)])
    row = tables(interim, "darwin_schedules").iloc[0]
    stops = list(row["stops"])
    assert [s["tpl"] for s in stops] == ["ORIG", "IP1", "PP1", "IP2", "DEST"]
    # pandas drops the zone of timestamps inside nested structs, the Parquet type keeps UTC
    assert pd.Timestamp(stops[-1]["wta"]) == pd.Timestamp("2026-10-15 23:40")  # 00:40 BST, the 16th
    assert row["origin_tpl"] == "ORIG" and row["dest_tpl"] == "DEST"


# incremental driver and status


def test_incremental_parse_skips_current_hour_and_finished_files(tmp_path):
    raw = tmp_path / "raw"
    write_gz(raw / "2026-10-15" / "10.jsonl.gz", fixture_lines())
    write_gz(raw / "2026-10-15" / "11.jsonl.gz", fixture_lines())
    interim = tmp_path / "interim"
    manifest = DarwinManifest(interim / "m.sqlite")
    now = datetime(2026, 10, 15, 11, 30, tzinfo=UTC)
    first = parse_directory(raw, interim, manifest, now=now)
    assert [r.key for r in first] == ["2026-10-15T10"]  # hour 11 is still being written
    assert parse_directory(raw, interim, manifest, now=now) == []
    later = parse_directory(raw, interim, manifest, now=datetime(2026, 10, 15, 12, 1, tzinfo=UTC))
    assert [r.key for r in later] == ["2026-10-15T11"]
    assert parse_directory(raw, interim, manifest, only_date=date(2026, 10, 14)) == []
    n, schedules, *_ = manifest.totals()
    assert (n, schedules) == (2, 4)
    manifest.close()


def test_file_that_grew_is_parsed_again(tmp_path):
    raw = tmp_path / "raw"
    path = raw / "2026-10-15" / "10.jsonl.gz"
    write_gz(path, fixture_lines())
    interim = tmp_path / "interim"
    manifest = DarwinManifest(interim / "m.sqlite")
    now = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
    parse_directory(raw, interim, manifest, now=now)
    with path.open("ab") as fh:  # a resume appended another gzip member
        fh.write(gzip.compress(line(99999, {"deactivated": {"rid": "202610150000001"}}) + b"\n"))
    assert len(parse_directory(raw, interim, manifest, now=now)) == 1
    manifest.close()


def test_status_lines_show_parse_totals_without_secrets(tmp_path):
    raw = tmp_path / "raw" / "darwin"
    write_gz(raw / "2026-10-15" / "10.jsonl.gz", fixture_lines())
    interim = tmp_path / "interim"
    manifest = DarwinManifest(interim / "darwin_manifest.sqlite")
    parse_directory(raw, interim, manifest, now=datetime(2026, 10, 15, 12, 0, tzinfo=UTC))
    manifest.close()
    text = "\n".join(darwin_status(raw, interim))
    assert "parsed: 1 files" in text and "schedules=2" in text
