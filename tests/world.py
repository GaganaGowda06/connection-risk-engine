"""A small synthetic world for live-state and API tests: one service on bhm-eus, a trained
bundle, history labels and Darwin-style events. Nothing here touches data/."""

import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from rail_risk import fixtures
from rail_risk.live import LiveStore

BANDS = fixtures.BANDS
TODAY = date(2026, 10, 15)
RID = "202610150000001"
TIPLOCS = {"BHAMNWS": "BHM", "EUSTON": "EUS", "ASTON": "AST", "MNCRPIC": "MAN"}
CFG = fixtures.CFG
snapshots = fixtures.snapshots
build_bundle = fixtures.build_bundle


def ts(hhmm: str, day: date = TODAY) -> pd.Timestamp:
    return pd.Timestamp(f"{day.isoformat()} {hhmm}", tz="UTC")


EVENT_COLUMNS = [
    "rid", "ssd", "tiploc", "event_type", "time_kind", "time", "planned_time", "late_reason",
    "message_ts", "partition", "offset",
]  # fmt: skip


def events_for(rid: str = RID, started: bool = True, offset0: int = 0) -> pd.DataFrame:
    """BHM dep 10:00, Coventry 10:30 to 10:32, Euston arr 11:30; two minutes late so far."""
    rows = []

    def add(tpl, etype, planned, actual=None, forecast=None, off=0):
        kind, when = ("actual", actual) if actual else ("forecast", forecast)
        rows.append(
            {
                "rid": rid, "ssd": pd.Timestamp(TODAY), "tiploc": tpl, "event_type": etype,
                "time_kind": kind, "time": ts(when) if when else pd.NaT,
                "planned_time": ts(planned), "late_reason": None,
                "message_ts": ts("10:40"), "partition": 0, "offset": offset0 + off,
            }
        )  # fmt: skip

    add("BHAMNWS", "dep", "10:00", actual="10:02" if started else None, forecast="10:02", off=1)
    add("COVNTRY", "arr", "10:30", actual="10:33" if started else None, forecast="10:33", off=2)
    add("COVNTRY", "dep", "10:32", actual="10:35" if started else None, forecast="10:35", off=3)
    add("EUSTON", "arr", "11:30", forecast="11:34", off=4)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def make_store(tmp: Path, events: pd.DataFrame, toc: str | None = "VT", newest: str = "10:43"):
    store = LiveStore(tmp / "interim", tmp / "raw", TIPLOCS)
    store.events = events
    if toc is not None:
        store.schedules[events["rid"].iloc[0]] = {
            "rid": events["rid"].iloc[0], "toc": toc, "cancel_reason": None, "n_stops": 4,
            "n_cancelled_stops": 0, "message_ts": ts("09:00"), "offset": 0,
        }  # fmt: skip
    store.newest_message_ts = ts(newest).to_pydatetime()
    return store


def write_raw_hour(path: Path, messages: list[dict], truncate_last: int = 0) -> None:
    """Hourly gz JSONL in the capture's line format, optionally cut mid-line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(m) + "\n" for m in messages).encode()
    data = gzip.compress(body)
    path.write_bytes(data[: len(data) - truncate_last] if truncate_last else data)


def ts_message(rid: str, partition: int, offset: int, tpl: str, planned: str, actual: str):
    return {
        "partition": partition, "offset": offset, "kafka_ts": 1760524800000 + offset,
        "payload": {
            "ts": "2026-10-15T11:41:00.000000+01:00",
            "uR": {
                "TS": [
                    {
                        "rid": rid, "uid": "A00001", "ssd": TODAY.isoformat(),
                        "Location": [
                            {"tpl": tpl, "ptd": planned, "dep": {"at": actual, "src": "TD"}},
                        ],
                    }
                ]
            },
        },
    }  # fmt: skip


def history_labels(spark):
    """28 days of destination delays for route bhm-eus at London hour 12 (11:30 UTC in October)."""
    rows = []
    for k in range(1, 29):
        d = TODAY - timedelta(days=k)
        for j in range(3):
            rows.append(("bhm-eus", d, 12, float(2 + (j + k) % 5)))
    return spark.createDataFrame(
        rows, "route_id string, service_date date, dest_hour int, delay_min double"
    )
