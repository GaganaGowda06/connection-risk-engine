"""Capture the Darwin Push Port Kafka feed to hourly gzip JSONL under data/raw/darwin.

Resumable: offsets are committed to the consumer group after each hourly file is closed and
on shutdown. A restart resumes from the committed offset; with no committed offset it starts at
the latest message. Anything older than the topic's retention is lost.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

from confluent_kafka import Consumer, KafkaError, KafkaException

from rail_risk.capture import GapDetector, HourlyWriter, strip_message
from rail_risk.config import Settings
from rail_risk.logging import configure_logging, get_logger

TOPIC = "prod-1010-Darwin-Train-Information-Push-Port-IIII2_0-JSON"
BOOTSTRAP = "pkc-z3p1v0.europe-west2.gcp.confluent.cloud:9092"
RAW_DIR = Path("data/raw/darwin")
FATAL = {
    KafkaError._AUTHENTICATION,
    KafkaError.TOPIC_AUTHORIZATION_FAILED,
    KafkaError.GROUP_AUTHORIZATION_FAILED,
    KafkaError.UNKNOWN_TOPIC_OR_PART,
}


def commit(consumer: Consumer, log) -> None:
    try:
        consumer.commit(asynchronous=False)
    except KafkaException as exc:
        # NO_OFFSET means nothing consumed since the last commit
        if exc.args[0].code() != KafkaError._NO_OFFSET:
            log.warning("commit_failed", error=exc.args[0].name())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=None, help="stop after this many hours")
    parser.add_argument("--minutes", type=float, default=None, help="stop after this many minutes")
    args = parser.parse_args()
    limit_s = None
    if args.hours is not None:
        limit_s = args.hours * 3600
    if args.minutes is not None:
        limit_s = args.minutes * 60

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

    def on_error(err: KafkaError) -> None:
        # librdkafka reconnects with backoff on its own; this only logs what it reports
        log.warning("kafka_client_error", code=err.name(), fatal=err.fatal())

    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "PLAIN",
            "sasl.username": settings.rdm_darwin_user.get_secret_value(),
            "sasl.password": settings.rdm_darwin_password.get_secret_value(),
            "group.id": settings.rdm_darwin_group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "reconnect.backoff.ms": 500,
            "reconnect.backoff.max.ms": 30000,
            "socket.keepalive.enable": True,
            "session.timeout.ms": 45000,
            "error_cb": on_error,
        }
    )
    writer = HourlyWriter(RAW_DIR)
    gaps = GapDetector(300)
    total_msgs = total_bytes = win_msgs = win_bytes = 0
    started = last_log = time.monotonic()
    exit_code = 0
    try:
        consumer.subscribe([TOPIC])
        log.info("capture_start", topic=TOPIC, limit_seconds=limit_s)
        while not stop and (limit_s is None or time.monotonic() - started < limit_s):
            msg = consumer.poll(1.0)
            if msg is not None:
                if msg.error():
                    code = msg.error().code()
                    if code in FATAL:
                        log.error("kafka_fatal", code=msg.error().name())
                        exit_code = 1
                        break
                    if code != KafkaError._PARTITION_EOF:
                        log.warning("kafka_error", code=msg.error().name())
                else:
                    _, ts_ms = msg.timestamp()
                    if ts_ms < 0:
                        ts_ms = int(time.time() * 1000)
                    gap = gaps.check(ts_ms)
                    if gap is not None:
                        log.warning("gap_detected", gap_seconds=round(gap), kafka_ts=ts_ms)
                    line = strip_message(msg.value(), msg.partition(), msg.offset(), ts_ms)
                    if writer.write(ts_ms, line):
                        writer.flush()
                        commit(consumer, log)
                    total_msgs += 1
                    win_msgs += 1
                    total_bytes += len(line)
                    win_bytes += len(line)
            now = time.monotonic()
            if now - last_log >= 60:
                writer.flush()
                log.info(
                    "throughput",
                    window_messages=win_msgs,
                    window_mb=round(win_bytes / 1e6, 2),
                    total_messages=total_msgs,
                    total_mb=round(total_bytes / 1e6, 2),
                    file=str(writer.path),
                )
                win_msgs = win_bytes = 0
                last_log = now
    finally:
        writer.close()
        commit(consumer, log)
        consumer.close()
        log.info(
            "capture_end",
            seconds=round(time.monotonic() - started),
            total_messages=total_msgs,
            total_mb=round(total_bytes / 1e6, 2),
            interrupted=stop,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
