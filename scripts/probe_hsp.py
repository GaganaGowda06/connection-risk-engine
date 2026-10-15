"""Minimal live probe of the HSP API on Rail Data Marketplace. Two calls, no loops."""

import json
import sys
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


def post(client: httpx.Client, endpoint: str, body: dict) -> httpx.Response:
    """POST with at most one retry, only on a timeout or a 5xx."""
    for attempt in (1, 2):
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


def save(name: str, day: date, resp: httpx.Response) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}_{day.isoformat()}.json"
    path.write_text(json.dumps(resp.json(), indent=2), encoding="utf-8")
    return path


def fail(endpoint: str, resp: httpx.Response) -> int:
    print(f"{endpoint}: HTTP {resp.status_code}")
    print(resp.text[:500])
    if resp.status_code == 404:
        print("Path not found: check the endpoint path against the RDM spec.")
    return 1


def main() -> int:
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
