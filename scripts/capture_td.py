#!/usr/bin/env python3
# ruff: noqa: E501
"""Capture the Network Rail Train Describer (TD_ALL_SIG_AREA) and TSR feeds, raw and unfiltered.

Every STOMP frame body is kept as received, one JSON line per frame
({"r": receive time ms, "t": topic, "m": message id, "b": body text}), in data/raw/td/YYYY-MM-DD/HH.jsonl.gz
(UTC receive hour). A durable subscription (client-id = login, fixed subscription names) lets a
reconnect resume where the broker held the messages. STOMP heartbeats are on, the reconnect backs
off, files are sync-flushed every few seconds. Log: counts per minute, gaps over 60 s, reconnects,
heartbeat misses. Credentials come from .env (NROD_EMAIL, NROD_PASSWORD) and are never logged.

    nohup ./scripts/run_td_capture.sh > /dev/null 2>&1 &
"""

import argparse
import gzip
import json
import os
import resource
import signal
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import stomp
from dotenv import dotenv_values

HOST, PORT = (
    "publicdatafeeds.networkrail.co.uk",
    61618,
)  # wiki "About the feeds", port confirmed open
TOPICS = {"TD_ALL_SIG_AREA": "crd-td", "TSR_ALL_ROUTE": "crd-tsr"}
HEARTBEAT_MS = 15000
FLUSH_S = 5.0
GAP_S = 60.0
STOP = threading.Event()


def now_ms() -> int:
    return int(time.time() * 1000)


def log(msg: str) -> None:
    print(f"{datetime.now(UTC).isoformat(timespec='seconds')} {msg}", flush=True)


class Writer:
    """One gzip member stream per UTC hour, sync-flushed; later appends add members."""

    def __init__(self, root: Path) -> None:
        self.root, self.hour, self.fh, self.last_flush = root, None, None, time.time()
        self.lock = threading.Lock()

    def write(self, recv_ms: int, topic: str, body: str, mid: str = "") -> None:
        t = datetime.fromtimestamp(recv_ms / 1000, UTC)
        key = (t.strftime("%Y-%m-%d"), t.strftime("%H"))
        rec = {"r": recv_ms, "t": topic, "m": mid, "b": body}
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        with self.lock:
            if key != self.hour:
                self._close()
                d = self.root / key[0]
                d.mkdir(parents=True, exist_ok=True)
                self.fh = gzip.open(d / f"{key[1]}.jsonl.gz", "ab")
                self.hour = key
            self.fh.write(line.encode())
            if time.time() - self.last_flush > FLUSH_S:
                self.fh.flush()
                self.last_flush = time.time()

    def flush(self) -> None:
        with self.lock:
            if self.fh:
                self.fh.flush()
                self.last_flush = time.time()

    def _close(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None

    def close(self) -> None:
        with self.lock:
            self._close()


class Listener(stomp.ConnectionListener):
    def __init__(self, writer: Writer, state: dict) -> None:
        self.w, self.s, self.conn = writer, state, None

    def on_message(self, frame) -> None:
        topic = frame.headers.get("destination", "").rsplit("/", 1)[-1]
        t = now_ms()
        gap = time.time() - self.s["last"]
        if gap > GAP_S:
            log(f"GAP ended: silent for {gap:.0f}s")
        self.s["gap_open"] = False
        self.w.write(t, topic, frame.body, frame.headers.get("message-id", ""))
        self.s["last"] = time.time()
        try:  # ack each message after it is written (ack: client-individual)
            self.conn.ack(frame.headers.get("ack") or frame.headers["message-id"])
        except Exception as exc:  # noqa: BLE001
            self.s["ack_fail"] += 1
            log(f"ack failed ({type(exc).__name__})")
        self.s["minute"][int(t // 60000)] += 1
        self.s["total"] += 1

    def on_error(self, frame) -> None:
        log(f"STOMP error frame: {str(frame.body)[:200]}")

    def on_disconnected(self) -> None:
        self.s["connected"] = False
        log("disconnected")

    def on_heartbeat_timeout(self) -> None:
        self.s["hb_miss"] += 1
        log("heartbeat timeout")


def connect(creds: dict, writer: Writer, state: dict) -> stomp.Connection12:
    conn = stomp.Connection12(
        [(HOST, PORT)],
        heartbeats=(HEARTBEAT_MS, HEARTBEAT_MS),
        keepalive=True,
        auto_decode=True,
        reconnect_attempts_max=1,
    )
    listener = Listener(writer, state)
    listener.conn = conn
    conn.set_listener("cap", listener)
    conn.connect(
        creds["NROD_EMAIL"],
        creds["NROD_PASSWORD"],
        wait=True,
        headers={"client-id": creds["NROD_EMAIL"]},
    )
    for i, (topic, name) in enumerate(TOPICS.items(), 1):
        conn.subscribe(
            f"/topic/{topic}",
            id=i,
            ack="client-individual",
            headers={"activemq.subscriptionName": name},
        )
    state["connected"] = True
    return conn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/td")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run)")
    args = ap.parse_args()
    env = {**dotenv_values(".env"), **os.environ}
    creds = {k: env.get(k) for k in ("NROD_EMAIL", "NROD_PASSWORD")}
    if not all(creds.values()):
        raise SystemExit("NROD_EMAIL and NROD_PASSWORD must be set in .env")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())
    writer = Writer(Path(args.out))
    state = {
        "minute": Counter(), "total": 0, "hb_miss": 0, "ack_fail": 0,
        "connected": False, "last": time.time(), "gap_open": False,
    }  # fmt: skip
    started, reconnects, backoff, conn = time.time(), 0, 1.0, None
    next_min = int(time.time() // 60)
    last_tick = time.time()
    while not STOP.is_set():
        if args.seconds and time.time() - started > args.seconds:
            break
        if not state["connected"]:
            try:
                if conn is not None:
                    try:
                        conn.disconnect()
                    except Exception:  # noqa: BLE001, S110
                        pass
                conn = connect(creds, writer, state)
                reconnects += 1
                log(f"connected (connection {reconnects}), subscribed {list(TOPICS)}")
                backoff = 1.0
            except Exception as exc:  # noqa: BLE001
                log(f"connect failed ({type(exc).__name__}); retry in {backoff:.0f}s")
                STOP.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
        STOP.wait(1.0)
        tick = time.time()
        if tick - last_tick > 10:
            log(
                f"STALL the loop did not run for {tick - last_tick:.0f}s (Mac asleep or suspended?)"
            )
        last_tick = tick
        if time.time() - writer.last_flush > FLUSH_S:
            writer.flush()
        silent = time.time() - state["last"]
        if silent > GAP_S and not state["gap_open"]:
            state["gap_open"] = True
            log(f"GAP open: silent for over {GAP_S:.0f}s, connected={state['connected']}")
        cur = int(time.time() // 60)
        while next_min < cur:
            n = state["minute"].pop(next_min, 0)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            m = datetime.fromtimestamp(next_min * 60, UTC).strftime("%H:%M")
            log(
                f"MINUTE {m} count={n} total={state['total']} hb_miss={state['hb_miss']} ack_fail={state['ack_fail']} peak_rss_mb={rss:.0f}"
            )
            next_min += 1
    writer.close()
    log(
        f"stopped after {time.time() - started:.0f}s, total={state['total']}, connections={reconnects}"
    )
    if conn is not None:
        try:
            conn.disconnect()
        except Exception:  # noqa: BLE001, S110
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
