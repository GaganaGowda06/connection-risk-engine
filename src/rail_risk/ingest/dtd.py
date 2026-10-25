"""Fetch the RDG timetable feed (DTD) from the National Rail Data Portal and archive it.

Token: POST /authenticate with username and password, the JSON reply carries "token".
Download: GET /api/staticfeeds/3.0/timetable with X-Auth-Token, which returns RJTTFnnn.ZIP.
Credentials come from the environment or .env (NRDP_EMAIL, NRDP_PASSWORD) and are never logged.
The feed is used under the NRE Developer Terms and Conditions v3.0 (OGL v2.0 based); the README
carries the required "Powered by National Rail Enquiries" credit.
"""

import csv
import hashlib
import os
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
from dotenv import dotenv_values

BASE = "https://opendata.nationalrail.co.uk"
AUTH_URL = f"{BASE}/authenticate"
TIMETABLE_URL = f"{BASE}/api/staticfeeds/3.0/timetable"
RAW_DIR = Path("data/raw/dtd")
MANIFEST = Path("data/manifests/dtd.csv")
MANIFEST_COLS = [
    "filename",
    "sequence",
    "last_modified",
    "msn_generated",
    "sha256",
    "bytes",
    "fetched_at",
]


class DtdError(RuntimeError):
    """Raised for any failure to authenticate, download or read the feed."""


@dataclass
class DtdFile:
    filename: str
    sequence: str
    last_modified: str
    msn_generated: str
    sha256: str
    bytes: int
    fetched_at: str


def credentials(env_file: Path = Path(".env")) -> tuple[str, str]:
    values = {**dotenv_values(env_file), **{k: v for k, v in os.environ.items() if k in _KEYS}}
    email, password = values.get("NRDP_EMAIL"), values.get("NRDP_PASSWORD")
    if not email or not password:
        raise DtdError("NRDP_EMAIL and NRDP_PASSWORD must be set in .env")
    return email, password


_KEYS = ("NRDP_EMAIL", "NRDP_PASSWORD")


def authenticate(client: httpx.Client, email: str, password: str) -> str:
    r = client.post(AUTH_URL, data={"username": email, "password": password})
    if r.status_code != 200:
        raise DtdError(f"NRDP authentication failed: HTTP {r.status_code}")
    token = r.json().get("token")
    if not token:
        raise DtdError("NRDP authentication reply has no token")
    return token


def download(client: httpx.Client, token: str, dest_dir: Path = RAW_DIR) -> tuple[Path, dict]:
    """Stream the timetable ZIP to dest_dir; returns the path and the response headers."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", TIMETABLE_URL, headers={"X-Auth-Token": token}) as r:
        if r.status_code in (401, 403):
            raise DtdError(f"NRDP rejected the token: HTTP {r.status_code}")
        if r.status_code != 200:
            raise DtdError(f"NRDP timetable download failed: HTTP {r.status_code}")
        disp = r.headers.get("content-disposition", "")
        m = re.search(r'filename="?([^";]+)"?', disp)
        name = Path(m.group(1)).name if m else ""
        if not re.fullmatch(r"RJTTF\d+\.ZIP", name, flags=re.IGNORECASE):
            raise DtdError(f"unexpected timetable filename {name!r}")
        tmp = dest_dir / (name + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
        headers = dict(r.headers)
    path = dest_dir / name
    tmp.replace(path)
    return path, headers


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_members(zip_path: Path) -> dict[str, bytes]:
    """The .MSN and .TSI members as bytes. A corrupt or truncated ZIP raises DtdError."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            bad = z.testzip()
            if bad:
                raise DtdError(f"corrupt member {bad} in {zip_path.name}")
            out = {}
            for info in z.infolist():
                ext = Path(info.filename).suffix.upper()
                if ext in (".MSN", ".TSI"):
                    out[ext] = z.read(info)
    except zipfile.BadZipFile as e:
        raise DtdError(f"{zip_path.name} is not a readable ZIP: {e}") from e
    for ext in (".MSN", ".TSI"):
        if ext not in out:
            raise DtdError(f"{zip_path.name} has no {ext} member")
    return out


def append_manifest(row: DtdFile, path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if new:
            w.writeheader()
        w.writerow(asdict(row))


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def last_modified_iso(headers: dict) -> str:
    raw = {k.lower(): v for k, v in headers.items()}.get("last-modified", "")
    if not raw:
        return ""
    return parsedate_to_datetime(raw).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
