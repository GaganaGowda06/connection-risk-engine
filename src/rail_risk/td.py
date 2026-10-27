"""Readers for the captured TD and TSR files (data/raw/td/YYYY-MM-DD/HH.jsonl.gz)."""

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

DAY_MS = 86_400_000
SKEW_MS = 3_600_000


def fix_time(time_ms: int, recv_ms: int) -> tuple[int, bool]:
    """Correct the cross-midnight bug: a TD timestamp about 24 hours off the receive time.

    Returns the timestamp in ms and whether it was corrected. Only an offset within an hour of
    plus or minus one day is corrected; anything else is left as sent.
    """
    off = time_ms - recv_ms
    for sign in (1, -1):
        if abs(off - sign * DAY_MS) <= SKEW_MS:
            return time_ms - sign * DAY_MS, True
    return time_ms, False


def frames(path: Path) -> Iterator[dict]:
    """Frames of one hourly file. The file being written has no gzip trailer yet, so a truncated
    tail ends the iteration quietly; a truncated last line is dropped."""
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    return
    except (EOFError, gzip.BadGzipFile):
        return


def messages(path: Path) -> Iterator[tuple[int, str, str, dict]]:
    """(receive ms, topic, message kind such as CA_MSG, message dict with time_ms fixed)."""
    for fr in frames(path):
        try:
            body = json.loads(fr["b"])
        except json.JSONDecodeError:
            continue
        for item in body if isinstance(body, list) else [body]:
            for kind, msg in item.items():
                if "time" in msg:
                    msg = {**msg, "time_ms": fix_time(int(msg["time"]), fr["r"])[0]}
                yield fr["r"], fr["t"], kind, msg
