"""Minimal live probe of the HSP API on Rail Data Marketplace. Two calls, no loops."""

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

from rail_risk.config import Settings

BASE = "https://api1.raildata.org.uk/1010-historical-service-performance-_hsp_1_1"
OUT_DIR = Path("data/probe")


def nearest_weekday(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


HTTP_ATTEMPTS = 0


def post(client: httpx.Client, endpoint: str, body: dict) -> httpx.Response:
    """POST with at most one retry, only on a timeout or a 5xx."""
    global HTTP_ATTEMPTS
    for attempt in (1, 2):
        HTTP_ATTEMPTS += 1
        try:
            resp = client.post(f"{BASE}/{endpoint}", json=body)
        except httpx.TimeoutException:
            if attempt == 2:
                raise
            continue
        if resp.status_code >= 500 and attempt == 1:
            continue
        return resp
    raise AssertionError("unreachable")


def save(name: str, day: date, resp: httpx.Response, suffix: str = "") -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}_{day.isoformat()}{suffix}.json"
    path.write_text(json.dumps(resp.json(), indent=2), encoding="utf-8")
    return path


def fail(endpoint: str, resp: httpx.Response) -> int:
    print(f"{endpoint}: HTTP {resp.status_code}")
    print(resp.text[:500])
    if resp.status_code == 404:
        print("Path not found: check the endpoint path against the RDM spec.")
    return 1


WINDOWS = [("0600", "1000"), ("1000", "1400"), ("1400", "1800"), ("1800", "2200")]
MAX_CALLS = 27  # 30 for the step, minus 3 attempts spent before this budget was enforced
MAX_DETAILS = 10


def recent_weekdays(count: int, min_age_days: int) -> list[date]:
    days: list[date] = []
    day = date.today() - timedelta(days=min_age_days)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    return days


def disruption_rank(service: dict) -> tuple[int, float]:
    attrs = service.get("serviceAttributesMetrics", {})
    metrics = service.get("Metrics") or []
    odd = (
        not metrics
        or not attrs.get("gbtt_pta")
        or int(attrs.get("matched_services") or 0) != len(attrs.get("rids") or [])
    )
    worst = min((float(m.get("percent_tolerance") or 0) for m in metrics), default=0.0)
    return (0 if odd else 1, worst)


def summarise_details(rid: str, payload: dict) -> str:
    locs = payload.get("serviceAttributesDetails", {}).get("locations", [])
    no_actual = [
        entry["location"]
        for entry in locs
        if (entry.get("gbtt_pta") and not entry.get("actual_ta"))
        or (entry.get("gbtt_ptd") and not entry.get("actual_td"))
    ]
    reasons = sorted({entry["late_canc_reason"] for entry in locs if entry.get("late_canc_reason")})
    return f"{rid}: stops={len(locs)} missing_actuals={no_actual} reasons={reasons}"


def find_cancellations(client: httpx.Client) -> int:
    candidates: list[tuple[tuple[int, float], date, str, str]] = []
    scanned = 0
    timeouts_in_row = 0
    for day in recent_weekdays(5, 7):
        for from_time, to_time in WINDOWS:
            cached = OUT_DIR / f"serviceMetrics_{day.isoformat()}_{from_time}-{to_time}.json"
            if cached.exists():
                payload = json.loads(cached.read_text(encoding="utf-8"))
                print(f"{day} {from_time}-{to_time}: cached")
            else:
                if HTTP_ATTEMPTS >= MAX_CALLS - MAX_DETAILS:
                    continue
                body = {
                    "from_loc": "BHM",
                    "to_loc": "EUS",
                    "from_time": from_time,
                    "to_time": to_time,
                    "from_date": day.isoformat(),
                    "to_date": day.isoformat(),
                    "days": "WEEKDAY",
                }
                try:
                    resp = post(client, "serviceMetrics", body)
                except httpx.TimeoutException:
                    timeouts_in_row += 1
                    print(f"{day} {from_time}-{to_time}: timed out twice, skipped")
                    if timeouts_in_row >= 2:
                        print("Two windows in a row timed out; stopping the scan.")
                        return finish(client, candidates, scanned)
                    continue
                timeouts_in_row = 0
                if 400 <= resp.status_code < 500:
                    return fail("serviceMetrics", resp)
                if resp.status_code != 200:
                    print(f"{day} {from_time}-{to_time}: HTTP {resp.status_code}, skipped")
                    continue
                save("serviceMetrics", day, resp, f"_{from_time}-{to_time}")
                payload = resp.json()
                time.sleep(1)
            services = payload.get("Services") or []
            scanned += len(services)
            print(f"{day} {from_time}-{to_time}: {len(services)} services")
            for svc in services:
                for rid in svc.get("serviceAttributesMetrics", {}).get("rids", []):
                    candidates.append((disruption_rank(svc), day, rid, from_time))
    return finish(client, candidates, scanned)


def finish(client: httpx.Client, candidates: list, scanned: int) -> int:
    print(f"http attempts so far: {HTTP_ATTEMPTS}, services scanned: {scanned}")
    # worst first, but spread across days so one bad day does not use all ten slots
    candidates.sort(key=lambda c: c[0])
    chosen: list[tuple[date, str]] = []
    per_day: dict[date, int] = {}
    for _, day, rid, _ in candidates:
        if per_day.get(day, 0) < 2 and (day, rid) not in chosen:
            chosen.append((day, rid))
            per_day[day] = per_day.get(day, 0) + 1
        if len(chosen) >= MAX_DETAILS:
            break
    for day, rid in chosen:
        if HTTP_ATTEMPTS >= MAX_CALLS:
            print("Call budget reached; stopping.")
            break
        try:
            resp = post(client, "serviceDetails", {"rid": rid})
        except httpx.TimeoutException:
            print(f"{rid}: timed out twice, skipped")
            continue
        if 400 <= resp.status_code < 500:
            return fail("serviceDetails", resp)
        if resp.status_code != 200:
            print(f"{rid}: HTTP {resp.status_code}, skipped")
            continue
        save("serviceDetails", day, resp, f"_{rid}")
        print(summarise_details(rid, resp.json()))
        time.sleep(1)
    print(f"http attempts this run: {HTTP_ATTEMPTS}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--find-cancellations", action="store_true")
    args = parser.parse_args()
    key = Settings().rdm_hsp_key
    if key is None:
        print("RAIL_RDM_HSP_KEY is not set; add it to .env and retry.")
        return 1
    secret = key.get_secret_value()
    bad = [i for i, c in enumerate(secret) if not c.isascii() or c.isspace()]
    if bad:
        print(f"RAIL_RDM_HSP_KEY has non-ASCII or whitespace characters at positions {bad}.")
        print("Re-copy the key from RDM into .env; no request was sent.")
        return 1
    day = nearest_weekday(date.today() - timedelta(days=14))
    headers = {"x-apikey": secret, "Content-Type": "application/json"}
    metrics_body = {
        "from_loc": "BHM",
        "to_loc": "EUS",
        "from_time": "0700",
        "to_time": "0800",
        "from_date": day.isoformat(),
        "to_date": day.isoformat(),
        "days": "WEEKDAY",
    }
    with httpx.Client(timeout=30, headers=headers) as client:
        if args.find_cancellations:
            return find_cancellations(client)
        resp = post(client, "serviceMetrics", metrics_body)
        if resp.status_code != 200:
            return fail("serviceMetrics", resp)
        print(f"serviceMetrics: HTTP 200, saved {save('serviceMetrics', day, resp)}")
        services = resp.json().get("Services") or []
        print(f"services returned: {len(services)}")
        rids = [
            rid
            for s in services
            for m in s.get("serviceAttributesMetrics", {}).get("rids", [])
            for rid in [m]
        ]
        if not rids:
            print("No rid found in serviceMetrics response; inspect the saved file.")
            return 1
        resp = post(client, "serviceDetails", {"rid": rids[0]})
        if resp.status_code != 200:
            return fail("serviceDetails", resp)
        print(f"serviceDetails: HTTP 200, saved {save('serviceDetails', day, resp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
