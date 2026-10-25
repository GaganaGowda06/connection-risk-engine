"""MSN and TSI parsing on lines in the RSPS5046 layout, plus the real feed when it is present."""

import math
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from rail_risk.ingest import dtd, mct

HEADER = "A" + " " * 29 + "FILE-SPEC=05 1.00 20/10/26 18.08.00   997"


def msn_record(name: str, tiploc: str, crs: str, minutes: str) -> str:
    """An 82-character station record: TIPLOC 37-43, CRS 50-52, MCT 64-65."""
    rec = list(" " * 82)

    def put(col: int, text: str) -> None:  # 1-based column
        rec[col - 1 : col - 1 + len(text)] = text

    put(1, "A")
    put(6, name)
    put(36, "3")
    put(37, tiploc)
    put(44, crs)
    put(50, crs)
    put(53, "14070")
    put(59, "62867")
    put(64, minutes)
    return "".join(rec)


def msn_bytes(records: list[str], header: bool = True, generated: str = "20/10/2026") -> bytes:
    head = ["/!! Start of file", "/!! Sequence:      995", f"/!! Generated:     {generated}"]
    body = ([HEADER.ljust(82)] if header else []) + [r.ljust(82) for r in records]
    return ("\r\n".join([*head, *body, "End of File"]) + "\r\n").encode("latin-1")


RECORDS = [
    msn_record("BIRMINGHAM NEW STREET", "BHAMNWS", "BHM", "12"),
    msn_record("BIRMINGHAM NEW STREET", "STADJN", "BHM", "12"),
    msn_record("MANCHESTER PICCADILLY", "MNCRPIC", "MAN", "10"),
    msn_record("BRISTOL TEMPLE MEADS", "BRSTLTM", "BRI", "10"),
    msn_record("BLANK PLACE", "BLNKPLC", "BLK", "  "),
    msn_record("SPLIT STATION", "SPLTONE", "SPL", " 3"),
    msn_record("SPLIT STATION", "SPLTTWO", "SPL", " 5"),
]
TSI = b"BHM,XC,VT,15,a comment, with a comma\r\nBHM,VT,XC,7,\r\n\r\n"


def test_msn_reads_the_three_hubs_and_skips_the_header():
    st = mct.parse_msn(msn_bytes(RECORDS))
    assert len(st) == len(RECORDS)  # the FILE-SPEC header is not a station
    got = st.groupby("crs")["minutes"].first()
    assert (got["BHM"], got["MAN"], got["BRI"]) == (12.0, 10.0, 10.0)
    assert st.loc[st["tiploc"] == "STADJN", "crs"].iloc[0] == "BHM"


def test_header_without_file_spec_05_fails_closed():
    with pytest.raises(mct.MctError, match="FILE-SPEC=05"):
        mct.parse_msn(msn_bytes(RECORDS, header=False))
    with pytest.raises(mct.MctError, match="Generated"):
        mct.parse_msn(msn_bytes(RECORDS, generated="soon"))


def test_short_or_non_numeric_records_fail_loudly():
    with pytest.raises(mct.MctError, match="short MSN"):
        short = msn_bytes(RECORDS).replace(b"End of File", b"A    SHORT STATION")
        mct.parse_msn(short)
    with pytest.raises(mct.MctError, match="not numeric"):
        mct.parse_msn(msn_bytes([msn_record("X", "XTIPLOC", "XXX", "1X")]))


def test_tsi_overrides_one_direction_only_and_keeps_comment_commas():
    t = mct.parse_tsi(TSI)
    assert list(t["minutes"]) == [15.0, 7.0]
    assert t["comment"].iloc[0] == "a comment, with a comma"
    table = mct.build_mct(msn_bytes(RECORDS), TSI, "RJTTF995.ZIP")
    assert mct.mct_lookup(table, "BHM", "XC", "VT") == 15.0  # TSI wins for this pair
    assert mct.mct_lookup(table, "BHM", "VT", "XC") == 7.0
    assert mct.mct_lookup(table, "BHM", "XC", "XC") == 12.0  # other pairs use MSN
    assert mct.mct_lookup(table, "BHM", "XC", "LM") == 12.0


def test_lookup_fails_closed_for_missing_blank_and_conflicting_stations():
    table = mct.build_mct(msn_bytes(RECORDS), TSI, "RJTTF995.ZIP")
    assert mct.mct_lookup(table, "ZZZ", "XC", "VT") is None  # station not in MSN
    assert mct.mct_lookup(table, "BLK", "XC", "VT") is None  # blank MCT, no default
    assert mct.mct_lookup(table, "SPL", "XC", "VT") is None  # records disagree: 3 and 5
    msn = table[(table["arr_toc"] == "*") & (table["dep_toc"] == "*")]
    assert math.isnan(msn.set_index("crs").loc["SPL", "minutes"])


def test_table_columns_and_provenance():
    table = mct.build_mct(msn_bytes(RECORDS), TSI, "RJTTF995.ZIP")
    assert list(table.columns) == mct.COLUMNS
    row = table[(table["crs"] == "MAN") & (table["arr_toc"] == "*")].iloc[0]
    assert (row["dep_toc"], row["source_file"], row["source_seq"]) == ("*", "RJTTF995.ZIP", "995")
    assert row["generated_date"] == "2026-10-20"


def test_missing_mct_file_is_a_clear_error(tmp_path):
    with pytest.raises(mct.MctError, match="not found"):
        mct.load_mct_table(tmp_path / "none.csv")


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def test_zip_reading_fails_on_corrupt_truncated_or_incomplete_files(tmp_path):
    good = make_zip(tmp_path / "RJTTF995.ZIP", {"A.MSN": msn_bytes(RECORDS), "A.TSI": TSI})
    assert set(dtd.read_members(good)) == {".MSN", ".TSI"}
    truncated = tmp_path / "RJTTF996.ZIP"
    truncated.write_bytes(good.read_bytes()[:-30])
    with pytest.raises(dtd.DtdError, match="not a readable ZIP"):
        dtd.read_members(truncated)
    garbage = tmp_path / "RJTTF997.ZIP"
    garbage.write_bytes(b"not a zip at all")
    with pytest.raises(dtd.DtdError, match="not a readable ZIP"):
        dtd.read_members(garbage)
    no_tsi = make_zip(tmp_path / "RJTTF998.ZIP", {"A.MSN": msn_bytes(RECORDS)})
    with pytest.raises(dtd.DtdError, match="no .TSI"):
        dtd.read_members(no_tsi)


def test_credentials_are_required_and_never_in_the_error(tmp_path, monkeypatch):
    monkeypatch.delenv("NRDP_EMAIL", raising=False)
    monkeypatch.delenv("NRDP_PASSWORD", raising=False)
    env = tmp_path / ".env"
    env.write_text("NRDP_EMAIL=someone@example.org\n")
    with pytest.raises(dtd.DtdError) as exc:
        dtd.credentials(env)
    assert "someone@example.org" not in str(exc.value)


REAL = Path("data/raw/dtd")


@pytest.mark.skipif(not list(REAL.glob("RJTTF*.ZIP")), reason="no downloaded timetable ZIP")
def test_real_feed_has_file_spec_05_and_the_three_hubs_parse():
    path = sorted(REAL.glob("RJTTF*.ZIP"))[-1]
    members = dtd.read_members(path)
    assert mct.parse_msn_header(members[".MSN"])["generated_date"]  # FILE-SPEC=05 present
    table = mct.build_mct(members[".MSN"], members[".TSI"], path.name)
    for hub in ("BHM", "MAN", "BRI"):
        value = mct.mct_lookup(table, hub, "XC", "VT")
        assert value is not None and 1 <= value <= 60, hub
    assert isinstance(table, pd.DataFrame)
