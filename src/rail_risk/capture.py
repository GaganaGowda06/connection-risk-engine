"""Pure helpers for the Darwin capture: payload stripping, hourly files, gap detection."""

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import IO


def strip_message(value: bytes | None, partition: int, offset: int, kafka_ts_ms: int) -> bytes:
    """One JSONL line: the Darwin payload plus partition, offset and Kafka timestamp.

    The Kafka value is a JMS wrapper whose `bytes` field holds the Darwin payload as a JSON
    string. If the wrapper or payload is not what we expect, the original text is kept under
    `raw` so that nothing is lost.
    """
    record: dict = {"partition": partition, "offset": offset, "kafka_ts": kafka_ts_ms}
    text = (value or b"").decode("utf-8", "replace")
    try:
        wrapper = json.loads(text)
        payload = json.loads(wrapper["bytes"])
        record["payload"] = payload
    except (ValueError, KeyError, TypeError):
        record["raw"] = text
    return json.dumps(record, separators=(",", ":")).encode("utf-8")


def hour_path(root: Path, kafka_ts_ms: int) -> Path:
    """data/raw/darwin/YYYY-MM-DD/HH.jsonl.gz for the UTC hour of a Kafka timestamp."""
    moment = datetime.fromtimestamp(kafka_ts_ms / 1000, tz=UTC)
    return root / moment.strftime("%Y-%m-%d") / f"{moment:%H}.jsonl.gz"


class HourlyWriter:
    """Append-only gzip JSONL, one file per UTC hour of the Kafka timestamp.

    The file hour never moves backwards: a late message from an earlier hour is written to the
    current file, so a file is closed once and offsets can be committed on each roll. Each
    open appends a new gzip member, which is valid gzip.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path: Path | None = None
        self._fh: IO[bytes] | None = None
        self._max_ts = 0

    def write(self, kafka_ts_ms: int, line: bytes) -> bool:
        """Write a line. Returns True when the previous file was closed (time to commit)."""
        self._max_ts = max(self._max_ts, kafka_ts_ms)
        target = hour_path(self.root, self._max_ts)
        rolled = False
        if target != self.path:
            rolled = self._fh is not None
            self.close()
            target.parent.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(target, "ab")
            self.path = target
        assert self._fh is not None
        self._fh.write(line + b"\n")
        return rolled

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class GapDetector:
    """Reports a jump between consecutive message timestamps above a threshold."""

    def __init__(self, threshold_s: float = 300.0) -> None:
        self.threshold_s = threshold_s
        self._last_ms: int | None = None

    def check(self, kafka_ts_ms: int) -> float | None:
        previous, self._last_ms = self._last_ms, kafka_ts_ms
        if previous is None:
            return None
        jump = (kafka_ts_ms - previous) / 1000
        return jump if jump > self.threshold_s else None
