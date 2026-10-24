from datetime import date
from pathlib import Path

import httpx
import pytest

from rail_risk import archive

DAY = date(2026, 10, 16)


def test_hour_path_and_urls():
    assert archive.hour_path(DAY, 1) == "2026/10/16/01.pport.gz"
    assert archive.day_url(DAY) == "https://ilovetrains.co.uk/darwin-push-port/2026-10-16"
    assert archive.download_url(DAY, 23) == (
        "https://ilovetrains.co.uk/api/download-darwin-dump?fileName=2026%2F10%2F16%2F23.pport.gz"
    )
    assert archive.local_path(Path("r"), DAY, 5) == Path("r/2026-10-16/05.pport.gz")


@pytest.mark.parametrize("hour", [-1, 24])
def test_hour_out_of_range(hour):
    with pytest.raises(ValueError):
        archive.hour_path(DAY, hour)


def test_parse_listing_reads_escaped_flight_data():
    html = (
        r"14:[{\"name\":\"/2026/10/16/00.pport.gz\",\"hour\":0,\"downloadUrl\":\"/api/x\","
        r"\"sizeBytes\":3036167},{\"name\":\"/2026/10/16/02.pport.gz\",\"hour\":2,"
        r"\"downloadUrl\":\"/api/y\",\"sizeBytes\":18144469}]"
    )
    assert archive.parse_listing(html) == {0: 3036167, 2: 18144469}
    assert archive.parse_listing("<html>no files</html>") == {}


def test_date_range_newest_first():
    days = archive.date_range_newest_first(date(2026, 2, 26), date(2026, 3, 1))
    assert days[0] == date(2026, 3, 1) and days[-1] == date(2026, 2, 26) and len(days) == 4
    with pytest.raises(ValueError):
        archive.date_range_newest_first(date(2026, 3, 1), date(2026, 2, 1))


def test_manifest_records_and_resumes(tmp_path):
    path = tmp_path / "m.sqlite"
    m = archive.Manifest(path)
    assert m.listing(DAY) is None and m.status(DAY, 3) is None
    m.save_listing(DAY, {0: 10, 3: 20})
    m.record(DAY, 3, "ok", 20, 20, "abc")
    m.record(DAY, 4, "gap")
    again = archive.Manifest(path)
    assert again.listing(DAY) == {0: 10, 3: 20}
    assert again.status(DAY, 3) == "ok"
    assert again.counts() == {"ok": (1, 20), "gap": (1, 0)}
    assert again.gaps() == [("2026-10-16", 4)]


def _fake_client(files: dict[str, bytes], listing: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/darwin-push-port"):
            return httpx.Response(200, text=listing)
        name = request.url.params["fileName"]
        return httpx.Response(200, content=files[name]) if name in files else httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _listing(sizes: dict[int, int]) -> str:
    items = ",".join(
        rf"{{\"name\":\"n\",\"hour\":{h},\"downloadUrl\":\"u\",\"sizeBytes\":{s}}}"
        for h, s in sizes.items()
    )
    return f"[{items}]"


def test_mirror_fetches_once_records_gaps_and_hashes(tmp_path):
    files = {archive.hour_path(DAY, 0): b"aaa", archive.hour_path(DAY, 1): b"bbbb"}
    client = _fake_client(files, _listing({0: 3, 1: 4}))
    m = archive.Manifest(tmp_path / "m.sqlite")
    log: list[str] = []
    res = archive.mirror(
        [DAY], tmp_path / "raw", m, client, pause=0, sleep=lambda s: None, log=log.append
    )
    assert (res.fetched, res.gaps, res.failed) == (2, 22, 0)
    assert (tmp_path / "raw/2026-10-16/00.pport.gz").read_bytes() == b"aaa"
    row = m.db.execute("SELECT bytes, sha256 FROM files WHERE hour=1").fetchone()
    assert row[0] == 4 and len(row[1]) == 64
    again = archive.mirror([DAY], tmp_path / "raw", m, client, pause=0, sleep=lambda s: None)
    assert (again.fetched, again.skipped) == (0, 24)


def test_mirror_retries_then_marks_failed_on_size_mismatch(tmp_path):
    client = _fake_client({archive.hour_path(DAY, 0): b"aaa"}, _listing({0: 99}))
    m = archive.Manifest(tmp_path / "m.sqlite")
    sleeps: list[float] = []
    res = archive.mirror(
        [DAY],
        tmp_path / "raw",
        m,
        client,
        pause=0,
        retries=3,
        backoff=1,
        sleep=sleeps.append,
        log=lambda s: None,
    )
    assert res.failed == 1 and m.status(DAY, 0) == "failed"
    assert [s for s in sleeps if s] == [1, 2, 4]
    assert not (tmp_path / "raw/2026-10-16/00.pport.gz").exists()
    assert not list((tmp_path / "raw").rglob("*.part"))


def test_mirror_stops_when_disk_is_below_the_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "free_bytes", lambda p: 10 * archive.GB)
    client = _fake_client({archive.hour_path(DAY, 0): b"aaa"}, _listing({0: 3}))
    m = archive.Manifest(tmp_path / "m.sqlite")
    res = archive.mirror(
        [DAY],
        tmp_path / "raw",
        m,
        client,
        pause=0,
        min_free_gb=60,
        sleep=lambda s: None,
        log=lambda s: None,
    )
    assert res.stopped_low_disk and res.fetched == 0 and m.status(DAY, 0) is None
