"""HTTP client for the risk API. The UI uses this and nothing else to reach the model."""

import os
import time
from typing import Any

import httpx

BASE_URL = os.environ.get("RAIL_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 5.0


class ApiDown(Exception):
    """The API could not be reached or answered with an unusable response."""


def sleep(seconds: float) -> None:
    """A hook so tests can skip the wait."""
    time.sleep(seconds)


def _request(method: str, path: str, **kw: Any) -> httpx.Response:
    try:
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT_SECONDS) as client:
            return client.request(method, path, **kw)
    except httpx.HTTPError as exc:
        raise ApiDown(str(exc)) from exc


def get_routes() -> list[dict[str, Any]]:
    r = _request("GET", "/routes")
    if r.status_code != 200:
        raise ApiDown(f"/routes answered {r.status_code}")
    return r.json()


def get_services(route: str) -> list[dict[str, Any]]:
    r = _request("GET", "/services", params={"route": route})
    if r.status_code != 200:
        raise ApiDown(f"/services answered {r.status_code}")
    return r.json()


def get_health() -> dict[str, Any]:
    r = _request("GET", "/health")
    if r.status_code != 200:
        raise ApiDown(f"/health answered {r.status_code}")
    return r.json()


def post_risk(rid: str, fare: float | None, ticket: str) -> tuple[int, dict[str, Any], int]:
    """(status code, body, Retry-After seconds). The body is {} when it is not JSON."""
    body: dict[str, Any] = {"rid": rid, "ticket": ticket}
    if fare is not None:
        body["fare"] = fare
    r = _request("POST", "/risk", json=body)
    try:
        data = r.json()
    except ValueError:
        data = {}
    return r.status_code, data, int(r.headers.get("retry-after", "2"))


def risk_with_retry(rid: str, fare: float | None, ticket: str) -> tuple[int, dict[str, Any]]:
    """Ask for the risk; if the service is warming, wait as told and ask once more."""
    status, data, retry_after = post_risk(rid, fare, ticket)
    if status == 202:
        sleep(min(max(retry_after, 1), 10))
        status, data, _ = post_risk(rid, fare, ticket)
    return status, data


def get_inbound() -> list[dict[str, Any]]:
    r = _request("GET", "/inbound")
    if r.status_code != 200:
        raise ApiDown(f"/inbound answered {r.status_code}")
    return r.json()


def post_connection(
    inbound_rid: str, outbound_departure: str | None, fare: float | None, ticket: str
) -> tuple[int, dict[str, Any], int]:
    """(status code, body, Retry-After seconds). The body is {} when it is not JSON."""
    body: dict[str, Any] = {"inbound_rid": inbound_rid, "ticket": ticket}
    if outbound_departure:
        body["outbound_departure"] = outbound_departure
    if fare is not None:
        body["fare"] = fare
    r = _request("POST", "/connection", json=body)
    try:
        data = r.json()
    except ValueError:
        data = {}
    return r.status_code, data, int(r.headers.get("retry-after", "2"))


def connection_with_retry(
    inbound_rid: str, outbound_departure: str | None, fare: float | None, ticket: str
) -> tuple[int, dict[str, Any]]:
    """Ask for the connection; if the service is warming, wait as told and ask once more."""
    status, data, retry_after = post_connection(inbound_rid, outbound_departure, fare, ticket)
    if status == 202:
        sleep(min(max(retry_after, 1), 10))
        status, data, _ = post_connection(inbound_rid, outbound_departure, fare, ticket)
    return status, data
