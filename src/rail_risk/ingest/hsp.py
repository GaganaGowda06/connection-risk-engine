"""Resumable HSP backfill: one-hour windows, write-once raw responses, a sqlite manifest."""

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rail_risk.config import Route
from rail_risk.ingest.hsp_parse import parse_details, to_frame, validate
from rail_risk.logging import get_logger

BASE = "https://api1.raildata.org.uk/1010-historical-service-performance-_hsp_1_1"


class RetryableHttpError(Exception):
    """429 or 5xx: worth another attempt."""


class HspHttpError(Exception):
    """A non-retryable HTTP error. The message never contains the API key."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


@dataclass
class Counters:
    metrics_calls: int = 0
    detail_calls: int = 0
    retries: int = 0
    failed_windows: int = 0
    windows_done: int = 0
    windows_skipped: int = 0
    services: int = 0
    rows: int = 0
    quarantined_rows: int = 0
    empty_details: int = 0
    reasons: dict[str, int] = field(default_factory=dict)


class HspClient:
    def __init__(
        self,
        api_key: str,
        *,
        http: httpx.Client | None = None,
        min_gap_s: float = 1.0,
        timeout_s: float = 60.0,
        base: str = BASE,
        retry_wait: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        counters: Counters | None = None,
    ) -> None:
        self._http = http or httpx.Client(timeout=timeout_s)
        self._headers = {"x-apikey": api_key, "Content-Type": "application/json"}
        self._base = base
        self._min_gap = min_gap_s
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self._retry_wait = retry_wait or wait_exponential(multiplier=2, min=2, max=30)
        self.counters = counters or Counters()

    def _once(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        if self._last_call is not None:
            wait = self._min_gap - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()
        if endpoint == "serviceMetrics":
            self.counters.metrics_calls += 1
        else:
            self.counters.detail_calls += 1
        resp = self._http.post(f"{self._base}/{endpoint}", json=body, headers=self._headers)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableHttpError(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise HspHttpError(resp.status_code, resp.text)
        return resp.json()

    def post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=self._retry_wait,
            retry=retry_if_exception_type((httpx.TimeoutException, RetryableHttpError)),
            reraise=True,
            sleep=self._sleep,
        )
        attempts = 0
        for attempt in retrying:
            with attempt:
                attempts += 1
                if attempts > 1:
                    self.counters.retries += 1
                return self._once(endpoint, body)
        raise AssertionError("unreachable")


class Manifest:
    """Completed windows and fetched rids. A rerun skips both."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS windows (
                route TEXT, day TEXT, hour INTEGER, services INTEGER, rows INTEGER,
                quarantined INTEGER, done_at TEXT, PRIMARY KEY (route, day, hour));
            CREATE TABLE IF NOT EXISTS rids (
                rid TEXT PRIMARY KEY, route TEXT, day TEXT, hour INTEGER);
            CREATE TABLE IF NOT EXISTS rid_routes (
                rid TEXT, route TEXT, PRIMARY KEY (rid, route));
            CREATE TABLE IF NOT EXISTS call_log (
                route TEXT, day TEXT, hour INTEGER, metrics_calls INTEGER, detail_calls INTEGER,
                retries INTEGER, failed INTEGER, at TEXT);
            INSERT OR IGNORE INTO rid_routes SELECT rid, route FROM rids;
            """
        )

    def window_done(self, route: str, day: date, hour: int) -> bool:
        cur = self._db.execute(
            "SELECT 1 FROM windows WHERE route=? AND day=? AND hour=?",
            (route, day.isoformat(), hour),
        )
        return cur.fetchone() is not None

    def has_rid(self, rid: str) -> bool:
        return self._db.execute("SELECT 1 FROM rids WHERE rid=?", (rid,)).fetchone() is not None

    def complete_window(
        self,
        route: str,
        day: date,
        hour: int,
        fresh: list[str],
        known: list[str],
        rows: int,
        quarantined: int,
    ) -> None:
        """Record a finished window. `fresh` rids were fetched here, `known` were fetched before."""
        with self._db:
            self._db.executemany(
                "INSERT OR IGNORE INTO rids VALUES (?,?,?,?)",
                [(r, route, day.isoformat(), hour) for r in fresh],
            )
            self._db.executemany(
                "INSERT OR IGNORE INTO rid_routes VALUES (?,?)",
                [(r, route) for r in fresh + known],
            )
            self._db.execute(
                "INSERT OR REPLACE INTO windows VALUES (?,?,?,?,?,?,datetime('now'))",
                (route, day.isoformat(), hour, len(fresh), rows, quarantined),
            )

    def log_calls(
        self, route: str, day: date, hour: int, before: "Counters", after: "Counters", failed: bool
    ) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO call_log VALUES (?,?,?,?,?,?,?,datetime('now'))",
                (
                    route,
                    day.isoformat(),
                    hour,
                    after.metrics_calls - before.metrics_calls,
                    after.detail_calls - before.detail_calls,
                    after.retries - before.retries,
                    int(failed),
                ),
            )

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        return self._db.execute(sql, params).fetchall()

    def close(self) -> None:
        self._db.close()


def write_once(path: Path, payload: dict[str, Any]) -> None:
    """Raw responses are never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except FileExistsError:
        pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_body(route: Route, day: date, hour: int) -> dict[str, str]:
    return {
        "from_loc": route.origin,
        "to_loc": route.destination,
        "from_time": f"{hour:02d}00",
        "to_time": f"{hour + 1:02d}00",
        "from_date": day.isoformat(),
        "to_date": day.isoformat(),
        "days": {5: "SATURDAY", 6: "SUNDAY"}.get(day.weekday(), "WEEKDAY"),
    }


def rids_from_metrics(payload: dict[str, Any]) -> list[str]:
    seen: dict[str, None] = {}
    for svc in payload.get("Services") or []:
        for rid in (svc.get("serviceAttributesMetrics") or {}).get("rids") or []:
            seen.setdefault(rid, None)
    return list(seen)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if "late_canc_reason" in df:  # keep the type stable when a window has no reason codes
        df = df.astype({"late_canc_reason": "string"})
    tmp = path.with_suffix(".parquet.tmp")  # readers never see a half-written file
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def run_window(
    route: Route,
    day: date,
    hour: int,
    client: HspClient,
    manifest: Manifest,
    *,
    raw_root: Path,
    interim_root: Path,
) -> None:
    c = client.counters
    log = get_logger("hsp")
    metrics_path = raw_root / "metrics" / day.isoformat() / f"{route.id}_{hour:02d}.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
    else:
        metrics = client.post("serviceMetrics", metrics_body(route, day, hour))
        write_once(metrics_path, metrics)

    rows: list[dict[str, Any]] = []
    fresh: list[str] = []
    known: list[str] = []
    for rid in rids_from_metrics(metrics):
        if manifest.has_rid(rid):
            known.append(rid)  # fetched and parsed already, possibly for another route
            continue
        details_path = raw_root / "details" / day.isoformat() / f"{rid}.json"
        if details_path.exists():
            payload = read_json(details_path)
        else:
            payload = client.post("serviceDetails", {"rid": rid})
            write_once(details_path, payload)
        fresh.append(rid)
        parsed = parse_details(payload)
        if not parsed:
            c.empty_details += 1
            log.warning("details_without_locations", rid=rid)
        rows.extend(parsed)

    good, bad = validate(to_frame(rows))
    if len(good):
        _write_parquet(
            good, interim_root / "hsp_stops" / f"date={day}" / f"{route.id}_{hour:02d}.parquet"
        )
    if len(bad):
        _write_parquet(
            bad, interim_root / "hsp_quarantine" / f"date={day}" / f"{route.id}_{hour:02d}.parquet"
        )
        for reason in bad.drop_duplicates("rid")["quarantine_reason"]:
            c.reasons[reason] = c.reasons.get(reason, 0) + 1
    manifest.complete_window(route.id, day, hour, fresh, known, len(good), len(bad))
    c.services += len(fresh)
    c.rows += len(good)
    c.quarantined_rows += len(bad)
    c.windows_done += 1


def backfill(
    route: Route,
    start: date,
    end: date,
    hours: range,
    client: HspClient,
    manifest: Manifest,
    *,
    raw_root: Path = Path("data/raw/hsp"),
    interim_root: Path = Path("data/interim"),
    newest_first: bool = False,
) -> Counters:
    """Run every unfinished window for route between start and end inclusive."""
    log = get_logger("hsp")
    c = client.counters
    span = (end - start).days
    days = [start + timedelta(days=i) for i in range(span + 1)]
    if newest_first:
        days.reverse()
    for day in days:
        for hour in hours:
            if manifest.window_done(route.id, day, hour):
                c.windows_skipped += 1
                continue
            before = replace(c)
            try:
                run_window(
                    route, day, hour, client, manifest, raw_root=raw_root, interim_root=interim_root
                )
            except (httpx.TimeoutException, RetryableHttpError) as exc:
                c.failed_windows += 1
                manifest.log_calls(route.id, day, hour, before, c, failed=True)
                log.error("window_failed", route=route.id, day=str(day), hour=hour, error=str(exc))
            except HspHttpError:
                manifest.log_calls(route.id, day, hour, before, c, failed=True)
                raise
            else:
                manifest.log_calls(route.id, day, hour, before, c, failed=False)
                log.info("window_done", route=route.id, day=str(day), hour=hour)
    return c


def rebuild_interim(manifest: Manifest, raw_root: Path, interim_root: Path) -> tuple[int, int]:
    """Re-parse saved raw details into the interim Parquet, with no API calls.

    Each rid is written under the (route, day, hour) window that first fetched it, so the files
    match the ones the original run produced. Returns (rows, quarantined_rows).
    """
    groups: dict[tuple[str, str, int], list[str]] = {}
    for rid, route, day, hour in manifest.query("SELECT rid, route, day, hour FROM rids"):
        groups.setdefault((route, day, hour), []).append(rid)
    total_rows = total_bad = 0
    for (route, day, hour), rids in sorted(groups.items()):
        rows: list[dict[str, Any]] = []
        for rid in sorted(rids):
            rows.extend(parse_details(read_json(raw_root / "details" / day / f"{rid}.json")))
        good, bad = validate(to_frame(rows))
        name = f"{route}_{hour:02d}.parquet"
        _write_parquet(good, interim_root / "hsp_stops" / f"date={day}" / name)
        if len(bad):
            _write_parquet(bad, interim_root / "hsp_quarantine" / f"date={day}" / name)
        total_rows += len(good)
        total_bad += len(bad)
    return total_rows, total_bad
