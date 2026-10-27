import gzip
import json

from rail_risk.td import DAY_MS, fix_time, frames, messages


def test_fix_time_corrects_a_day_off_only():
    recv = 1_793_085_584_683
    assert fix_time(recv - 1000, recv) == (recv - 1000, False)
    assert fix_time(recv + DAY_MS + 5000, recv) == (recv + 5000, True)
    assert fix_time(recv - DAY_MS - 5000, recv) == (recv - 5000, True)
    assert fix_time(recv + 3 * 3_600_000, recv)[1] is False


def _write(path, rows, close=True):
    fh = gzip.open(path, "wb")
    for r in rows:
        fh.write((json.dumps(r) + "\n").encode())
    if close:
        fh.close()
    else:
        fh.flush()
    return fh


def test_messages_unpack_frames_and_fix_time(tmp_path):
    recv = 1_793_085_584_683
    body = [
        {
            "CA_MSG": {
                "area_id": "NK",
                "time": str(recv + DAY_MS),
                "from": "0106",
                "to": "0104",
                "descr": "2T59",
            }
        },
        {"SF_MSG": {"area_id": "Q5", "time": str(recv), "address": "54", "data": "FE"}},
    ]
    p = tmp_path / "07.jsonl.gz"
    _write(p, [{"r": recv, "t": "TD_ALL_SIG_AREA", "b": json.dumps(body)}])
    out = list(messages(p))
    assert [k for _, _, k, _ in out] == ["CA_MSG", "SF_MSG"]
    assert out[0][3]["time_ms"] == recv


def test_open_file_without_trailer_is_readable(tmp_path):
    p = tmp_path / "08.jsonl.gz"
    fh = gzip.open(p, "ab")
    fh.write(b'{"r":1,"t":"TSR_ALL_ROUTE","b":"[]"}\n')
    fh.flush()
    assert len(list(frames(p))) == 1
    fh.close()
