"""Capture the Darwin Push Port Kafka feed to hourly gzip JSONL under data/raw/darwin."""

import argparse
import gzip
import json
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from confluent_kafka import Consumer, KafkaError

from rail_risk.config import Settings
from rail_risk.logging import configure_logging, get_logger

TOPIC = "prod-1010-Darwin-Train-Information-Push-Port-IIII2_0-JSON"
BOOTSTRAP = "pkc-z3p1v0.europe-west2.gcp.confluent.cloud:9092"
RAW_DIR = Path("data/raw/darwin")
FATAL = {
    KafkaError._AUTHENTICATION,
    KafkaError._ALL_BROKERS_DOWN,
    KafkaError.TOPIC_AUTHORIZATION_FAILED,
    KafkaError.GROUP_AUTHORIZATION_FAILED,
    KafkaError.UNKNOWN_TOPIC_OR_PART,
}


class HourlyWriter:
    """Append-only gzip JSONL, one file per UTC hour. Each append adds a gzip member."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._hour: str | None = None
        self._fh: IO[bytes] | None = None

    def write(self, now: datetime, line: bytes) -> None:
        hour = now.strftime("%Y-%m-%d/%H")
        if hour != self._hour:
            self.close()
            path = self.root / f"{hour}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(path, "ab")
            self._hour = hour
        assert self._fh is not None
        self._fh.write(line + b"\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def envelope(msg, now: datetime) -> bytes:
    value = msg.value() or b""
    try:
        body = json.loads(value)
    except ValueError:
        body = {"unparsed": value.decode("utf-8", "replace")}
    record = {
        "received_at": now.isoformat(),
        "partition": msg.partition(),
        "offset": msg.offset(),
        "value": body,
    }
    return json.dumps(record, separators=(",", ":")).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=10)
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level, settings.log_json)
    log = get_logger("capture_darwin")
    if not (
        settings.rdm_darwin_user and settings.rdm_darwin_password and settings.rdm_darwin_group
    ):
        log.error(
            "darwin_credentials_missing",
            need="RAIL_RDM_DARWIN_USER, RAIL_RDM_DARWIN_PASSWORD, RAIL_RDM_DARWIN_GROUP",
        )
        return 1

    stop = False

    def request_stop(signum, frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "PLAIN",
            "sasl.username": settings.rdm_darwin_user.get_secret_value(),
            "sasl.password": settings.rdm_darwin_password.get_secret_value(),
            "group.id": settings.rdm_darwin_group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        }
    )
    writer = HourlyWriter(RAW_DIR)
    total_msgs = total_bytes = win_msgs = win_bytes = 0
    started = last_log = time.monotonic()
    deadline = started + args.minutes * 60
    exit_code = 0
    try:
        consumer.subscribe([TOPIC])
        log.info("capture_start", topic=TOPIC, minutes=args.minutes)
        while not stop and time.monotonic() < deadline:
            msg = consumer.poll(1.0)
            if msg is not None:
                if msg.error():
                    code = msg.error().code()
                    if code in FATAL:
                        log.error("kafka_fatal", code=str(msg.error().name()))
                        exit_code = 1
                        break
                    if code != KafkaError._PARTITION_EOF:
                        log.warning("kafka_error", code=str(msg.error().name()))
                else:
                    size = len(msg.value() or b"")
                    writer.write(datetime.now(UTC), envelope(msg, datetime.now(UTC)))
                    total_msgs += 1
                    win_msgs += 1
                    total_bytes += size
                    win_bytes += size
            now = time.monotonic()
            if now - last_log >= 60:
                log.info(
                    "throughput",
                    window_messages=win_msgs,
                    window_mb=round(win_bytes / 1e6, 2),
                    total_messages=total_msgs,
                    total_mb=round(total_bytes / 1e6, 2),
                )
                win_msgs = win_bytes = 0
                last_log = now
    finally:
        writer.close()
        consumer.close()
        elapsed = time.monotonic() - started
        log.info(
            "capture_end",
            seconds=round(elapsed),
            total_messages=total_msgs,
            total_mb=round(total_bytes / 1e6, 2),
            interrupted=stop,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
