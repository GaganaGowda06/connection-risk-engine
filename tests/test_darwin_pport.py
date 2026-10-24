"""Parser tests on small hand-written messages that follow the v18 layout seen in the archive."""

import gzip
import json
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from rail_risk.ingest import darwin_pport as dp
from rail_risk.ingest.pport_extract import split_counts, stop_actuals_for_day, train_pool_table

HEAD = (
    '<?xml version="1.0" encoding="utf-8"?><Pport '
    'xmlns:fc="http://www.thalesgroup.com/rtti/PushPort/Forecasts/v4" '
    'xmlns:s5="http://www.thalesgroup.com/rtti/PushPort/Schedules/v5" '
    'ts="{ts}" version="18.0" xmlns="http://www.thalesgroup.com/rtti/PushPort/v18">'
)


def msg(ts: str, body: str) -> str:
    return HEAD.format(ts=ts) + f'<uR updateOrigin="TD">{body}</uR></Pport>\n'


def write_hour(path: Path, docs: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write("".join(docs))
    return path


SCHEDULE = (
    '<schedule rid="202610187000001" uid="A00001" trainId="1A00" ssd="2026-10-18" toc="XX" '
    'status="P" trainCat="XX"><s5:OR tpl="AAAAAAA" act="TB" ptd="23:50" wtd="23:50" />'
    '<s5:IP tpl="BBBBBBB" act="T " pta="00:10" ptd="00:11" wta="00:10" wtd="00:11" />'
    '<s5:DT tpl="CCCCCCC" act="TF" pta="00:30" wta="00:30" /></schedule>'
)
TS = (
    '<TS rid="202610187000001" uid="A00001" ssd="2026-10-18"><fc:Location tpl="BBBBBBB" '
    'wta="00:10" wtd="00:11" pta="00:10" ptd="00:11"><fc:arr at="00:12" atClass="Automatic" '
    'src="TD" /><fc:dep et="00:13" src="Darwin" /><fc:plat>2</fc:plat></fc:Location></TS>'
)


# ---------------------------------------------------------------- time


def test_message_time_with_seven_digit_fraction_and_offset():
    us, ordinal, secs, off = dp.parse_message_time("2026-10-16T08:00:00.0514381+01:00")
    assert off == 3600 and secs == 8 * 3600
    assert (
        us
        == (date(2026, 10, 16).toordinal() - dp.EPOCH_ORD) * 86400 * 10**6
        + 7 * 3600 * 10**6
        + 51438
    )


def test_local_to_utc_summer_winter_and_change_days():
    d = date(2026, 7, 1).toordinal()
    assert dp.local_to_utc(d, 12 * 3600) == [(d - dp.EPOCH_ORD) * 86400 + 11 * 3600]
    w = date(2026, 1, 5).toordinal()
    assert dp.local_to_utc(w, 12 * 3600) == [(w - dp.EPOCH_ORD) * 86400 + 12 * 3600]
    autumn = date(2025, 10, 26).toordinal()  # clocks go back: 01:30 happens twice
    two = dp.local_to_utc(autumn, 90 * 60)
    assert len(two) == 2 and two[1] - two[0] == 3600
    spring = date(2026, 3, 29).toordinal()  # 01:30 does not exist
    assert len(dp.local_to_utc(spring, 90 * 60)) <= 2
    after = dp.local_to_utc(spring, 3 * 3600)
    assert after == [(spring - dp.EPOCH_ORD) * 86400 + 2 * 3600]


def test_schedule_times_roll_over_midnight_and_pick_the_later_repeated_hour():
    ssd = date(2026, 10, 18).toordinal()
    a, b, c = dp.schedule_times(ssd, ["23:50", "00:10", "00:30"])
    assert b - a == 20 * 60 and c - b == 20 * 60
    night = date(2025, 10, 26).toordinal()
    first, second = dp.schedule_times(night, ["01:50", "01:10"])
    assert second - first == 20 * 60  # the repeated hour: 01:10 the second time is 20 minutes on


def test_early_forecast_lands_on_the_service_date_not_the_message_day():
    ssd = date(2026, 10, 14).toordinal()
    msg_utc = (date(2026, 10, 13).toordinal() - dp.EPOCH_ORD) * 86400 + 2 * 3600
    (placed,) = dp.service_day_times(ssd, ["00:40"], msg_utc)
    assert placed == (ssd - dp.EPOCH_ORD) * 86400 + 40 * 60 - 3600  # 00:40 BST on the 14th
    late_msg = (ssd + 1 - dp.EPOCH_ORD) * 86400 + 1200
    (after,) = dp.service_day_times(ssd, ["00:40"], late_msg)
    assert after == placed + 86400


def test_split_for_boundaries():
    cfg = dp.load_stage1_config()
    assert dp.split_for(date(2026, 10, 16), cfg) == "train_pool"
    assert dp.split_for(date(2026, 10, 17), cfg) == "oot_week"
    assert dp.split_for(date(2026, 10, 23), cfg) == "oot_week"
    assert dp.split_for(date(2026, 10, 24), cfg) == "train_pool"
    assert dp.split_for(date(2026, 2, 1), cfg) == "winter_test"
    assert dp.split_for(date(2026, 2, 28), cfg) == "winter_test"
    assert dp.split_for(date(2026, 3, 1), cfg) == "train_pool"


def test_documents_split_even_when_a_message_spans_lines():
    raw = b'<?xml version="1.0"?><a>\n x \n</a>\n<?xml version="1.0"?><b/>\n'
    import io

    docs = list(dp.iter_documents(io.BytesIO(raw)))
    assert len(docs) == 2 and b"x" in docs[0]


# ---------------------------------------------------------------- one file


def test_parse_file_writes_tables_split_and_source_hash(tmp_path):
    src = write_hour(
        tmp_path / "raw/2026-10-18/00.pport.gz",
        [
            msg("2026-10-18T00:05:00.1+01:00", SCHEDULE),
            msg("2026-10-18T00:12:30.2+01:00", TS),
            msg("2026-10-18T00:40:00.0+01:00", '<deactivated rid="202610187000001" />'),
        ],
    )
    out = tmp_path / "out"
    st = dp.parse_file(src, out)
    assert st.status == "ok" and st.documents == 3 and st.parse_errors == 0
    sched = pq.read_table(out / "schedule_points/service_date=2026-10-18").to_pandas()
    assert list(sched["tpl"]) == ["AAAAAAA", "BBBBBBB", "CCCCCCC"]
    assert sched["wt_dep"].iloc[0] < sched["wt_arr"].iloc[1] < sched["wt_arr"].iloc[2]
    assert set(sched["split"]) == {"oot_week"}
    upd = pq.read_table(out / "stop_updates/service_date=2026-10-18").to_pandas()
    arr = upd[upd["event"] == "arr"].iloc[0]
    assert (arr["actual"] - arr["wt_arr"]).total_seconds() == 120
    assert upd[upd["event"] == "dep"].iloc[0]["est"] > arr["actual"]
    part = next((out / "stop_updates/service_date=2026-10-18").glob("*.parquet"))
    meta = pq.read_schema(part).metadata
    assert meta[b"source_sha256"].decode() == st.source_sha256 == dp.sha256_file(src)
    assert dp.parse_file(src, out).documents == 3  # second call reads the stats file
    assert split_counts(out)["stop_updates"] == {"oot_week": len(upd)}
    assert train_pool_table(out, "stop_updates").num_rows == 0


def test_stop_actuals_takes_the_last_actual_and_drops_removed(tmp_path):
    first = msg("2026-10-18T00:12:30+01:00", TS)
    removed = msg(
        "2026-10-18T00:20:00+01:00",
        TS.replace(
            '<fc:arr at="00:12" atClass="Automatic" src="TD" />', '<fc:arr atRemoved="true" />'
        ),
    )
    src = write_hour(tmp_path / "raw/2026-10-18/00.pport.gz", [first, removed])
    out = tmp_path / "out"
    dp.parse_file(src, out)
    n = stop_actuals_for_day(out, date(2026, 10, 18))
    table = pq.read_table(out / "stop_actuals/service_date=2026-10-18").to_pandas()
    assert n == len(table) == 0 or set(table["event"]) <= {"dep", "pass"}
    assert "arr" not in set(table["event"])


def test_missing_file_is_logged_and_not_cached(tmp_path):
    st = dp.parse_file(tmp_path / "raw/2026-10-20/10.pport.gz", tmp_path / "out")
    assert st.status == "missing"
    assert not list((tmp_path / "out/_stats").glob("*"))


def test_truncated_gzip_keeps_rows_read_and_continues(tmp_path):
    docs = [
        msg(f"2026-10-16T08:{i:02d}:00+01:00", TS.replace("2026-10-18", "2026-10-16"))
        for i in range(50)
    ]
    full = write_hour(tmp_path / "raw/2026-10-16/08.pport.gz", docs)
    data = full.read_bytes()
    cut = tmp_path / "raw/2026-10-16/09.pport.gz"
    cut.write_bytes(data[: int(len(data) * 0.6)])
    st = dp.parse_file(cut, tmp_path / "out")
    assert st.status == "truncated" and st.error
    json.loads((tmp_path / "out/_stats/2026-10-16_09.json").read_text())
    good = dp.parse_file(full, tmp_path / "out")
    assert good.status == "ok" and good.documents == 50


def test_malformed_message_is_counted_and_skipped(tmp_path):
    bad = '<?xml version="1.0"?><Pport ts="2026-10-16T08:00:00+01:00"><uR><TS rid="1"'
    src = write_hour(
        tmp_path / "raw/2026-10-16/08.pport.gz",
        [msg("2026-10-16T08:00:00+01:00", TS.replace("2026-10-18", "2026-10-16")), bad + "\n"],
    )
    st = dp.parse_file(src, tmp_path / "out")
    assert st.status == "ok" and st.parse_errors == 1 and st.rows["stop_updates"] >= 1
