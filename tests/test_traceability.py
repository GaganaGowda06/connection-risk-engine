# ruff: noqa: E501
"""GOAL test 8 for the network service: every numeric or time field in a response has a source.

Runs against the committed sample ledger (data/sample/), so it works in CI and on a fresh clone.
The break tests damage a copy of the ledger and check that the service refuses or says why.
"""

import json
import re
import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rail_risk.serving import app as A
from rail_risk.serving import artifacts as AR
from rail_risk.serving import provenance as PV

SAMPLE = Path("data/sample/state_oot/state.sqlite")
TIME = re.compile(r"^\d{4}-\d{2}-\d{2}")
MONEY = re.compile(r"fare|price|payout|repay|compensation|refund|pound", re.I)
RID, OUT = "202610176700683", "202610176742156"


def _client(tmp_path, monkeypatch, mct_table=None, edit=None):
    state = tmp_path / "state"
    state.mkdir()
    shutil.copy(SAMPLE, state / "state.sqlite")
    if edit:
        con = sqlite3.connect(state / "state.sqlite")
        edit(con)
        con.commit()
        con.close()
    monkeypatch.setattr(A, "STATE_DIR", state)
    monkeypatch.setattr(A, "_LOCAL", type(A._LOCAL)())  # no connection kept from another test
    monkeypatch.setattr(PV, "MCT_TABLE", mct_table or tmp_path / "no_such_mct.csv")
    PV._TABLE_CACHE.clear()
    return TestClient(A.app)


@pytest.fixture
def client(tmp_path, monkeypatch):
    return _client(tmp_path, monkeypatch)


def _leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if not path and k == "provenance":
                continue
            yield from _leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v, f"{path}[]")
    else:
        yield re.sub(r"(\[\])+$", "", path) if isinstance(x, (int, float, str)) else path, x


def _keys(x, out=None):
    out = set() if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            out.add(k)
            _keys(v, out)
    elif isinstance(x, list):
        for v in x:
            _keys(v, out)
    return out


def _numeric_or_time(v) -> bool:
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, float)) or (isinstance(v, str) and bool(TIME.match(v)))


def _responses(client):
    out = {"health": client.get("/health").json()}
    out["risk"] = client.post("/risk", json={"rid": _sample_rid()}).json()
    out["connection"] = client.post("/connection", json={"in_rid": RID, "out_rid": OUT}).json()
    hub, at = _sample_inbound()
    out["inbound"] = client.get("/inbound", params={"hub": hub, "at": at}).json()
    return out


def _sample_rid() -> str:
    con = sqlite3.connect(f"file:{SAMPLE}?mode=ro", uri=True)
    try:
        return con.execute("select rid from arrival limit 1").fetchone()[0]
    finally:
        con.close()


def _sample_inbound() -> tuple[str, str]:
    con = sqlite3.connect(f"file:{SAMPLE}?mode=ro", uri=True)
    try:
        t = con.execute("select min(plan_arr_in) from pair where hub='BHM'").fetchone()[0]
    finally:
        con.close()
    return "BHM", t[:16]


def test_every_numeric_field_has_a_provenance_entry(client):
    rs = _responses(client)
    assert all("provenance" in r for r in rs.values())
    seen, distinct = 0, set()
    for name, r in rs.items():
        fields = r["provenance"]["fields"]
        for path, v in _leaves(r):
            if _numeric_or_time(v):
                seen += 1
                distinct.add((name, path))
                assert path in fields, f"{name}: {path} = {v!r} has no provenance entry"
        for path, entry in fields.items():
            key = entry.split(":")[0]
            assert key in r["provenance"]["sources"], f"{name}: {path} names unknown source {key}"
    assert seen > 100  # the inbound answer was not empty
    assert len(distinct) == 41, "README and reports/stage7_audit.md say 41 fields; update both"


def test_responses_carry_run_ids_freshness_and_mct_source(client):
    rs = _responses(client)
    for r in rs.values():
        assert r["ids"]["arrival_model"] == AR.ARRIVAL_RUN
        assert r["ids"]["arrival_calibration"] == AR.CALIBRATION_RUN
        assert r["ids"]["pmiss_model"] == AR.PMISS_RUN
        fresh = r.get("freshness", r)  # /health carries the freshness fields at the top
        assert fresh["data_as_of"] and "stale" in fresh
    mct = rs["connection"]["provenance"]["sources"]["mct"]
    assert mct["file"].startswith("RJTTF") and isinstance(mct["sequence"], int)
    assert mct["generated"]
    assert rs["health"]["status"] == "ok" and rs["health"]["problems"] == []
    assert rs["health"]["provenance"]["checks"]["ledger_runs_match_service"] is True


def test_the_service_prices_nothing(client):
    for r in _responses(client).values():
        bad = [k for k in _keys(r) if MONEY.search(k)]
        assert not bad, f"money-like fields in a response: {bad}"
    for f in Path("src/rail_risk/serving").glob("*.py"):
        imports = re.findall(r"^\s*(?:from|import)\s+\S*repay", f.read_text(), re.M)
        assert not imports, f"{f.name} imports the Delay Repay module"


def test_missing_mct_refuses_a_connection_answer_and_shows_the_reason(tmp_path, monkeypatch):
    def drop(con):
        con.execute("delete from meta where k='provenance'")

    c = _client(tmp_path, monkeypatch, edit=drop)  # no stamp and no data/ref/mct.csv
    r = c.post("/connection", json={"in_rid": RID, "out_rid": OUT})
    assert r.status_code == 503 and "mct_min has no source" in r.json()["detail"]
    h = c.get("/health").json()
    assert h["status"] == "degraded" and any("no MCT source" in p for p in h["problems"])
    assert c.post("/risk", json={"rid": _sample_rid()}).status_code == 200  # needs no MCT


def test_unrecorded_ledger_runs_are_shown_not_hidden(tmp_path, monkeypatch):
    table = tmp_path / "mct.csv"
    table.write_text(
        "crs,arr_toc,dep_toc,minutes,source_file,source_seq,generated_date\n"
        "BHM,*,*,5.0,RJTTF999.ZIP,999,2026-10-20\n"
    )

    def drop(con):
        con.execute("delete from meta where k='provenance'")

    c = _client(tmp_path, monkeypatch, mct_table=table, edit=drop)
    r = c.post("/connection", json={"in_rid": RID, "out_rid": OUT}).json()
    assert r["provenance"]["sources"]["mct"]["sequence"] == 999
    assert r["provenance"]["sources"]["ledger_runs"]["matches_service"] is None


def test_a_run_id_that_does_not_match_the_loaded_model_is_refused(tmp_path, monkeypatch):
    def swap(con):
        v = json.loads(con.execute("select v from meta where k='provenance'").fetchone()[0])
        v["runs"]["pmiss_model"] = "0" * 32
        con.execute("update meta set v=? where k='provenance'", (json.dumps(v),))

    c = _client(tmp_path, monkeypatch, edit=swap)
    for resp in (
        c.post("/risk", json={"rid": _sample_rid()}),
        c.post("/connection", json={"in_rid": RID, "out_rid": OUT}),
        c.get("/inbound", params={"hub": "BHM"}),
    ):
        assert resp.status_code == 503 and "pmiss_model" in resp.json()["detail"]
    h = c.get("/health").json()
    assert h["status"] == "degraded" and any("pmiss_model" in p for p in h["problems"])


def test_a_stale_feed_is_flagged_with_its_reason(tmp_path, monkeypatch):
    def age(con):
        f = json.loads(con.execute("select v from meta where k='fresh'").fetchone()[0])
        f["last_minute"] -= 180
        con.execute("update meta set v=? where k='fresh'", (json.dumps(f),))

    c = _client(tmp_path, monkeypatch, edit=age)
    r = c.post("/risk", json={"rid": _sample_rid()}).json()
    assert r["freshness"]["stale"] is True and re.fullmatch(
        r"no message for 1[89]\d min", r["freshness"]["stale_reason"]
    )
    h = c.get("/health").json()
    assert h["status"] == "degraded" and h["stale"] is True


def test_engine_start_fails_without_the_mct_table(tmp_path):
    from rail_risk.ingest import mct

    with pytest.raises(mct.MctError, match="no MCT value is assumed"):
        mct.load_mct_table(tmp_path / "missing.csv")
