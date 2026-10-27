#!/usr/bin/env python3
# ruff: noqa: E501
"""Download the NROD static files SMART and CORPUS with the NROD login from .env.

Saves the raw download under data/raw/nrod/ (never edited), appends date, file, bytes and sha256 to
data/manifests/nrod.csv. Credentials are read from .env and never printed.

    uv run python scripts/fetch_nrod_static.py
"""

import csv
import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import dotenv_values

BASE = "https://publicdatafeeds.networkrail.co.uk/ntrod/SupportingFileAuthenticate"
FILES = {"CORPUS": "CORPUSExtract.json.gz", "SMART": "SMARTExtract.json.gz"}


def main() -> None:
    env = {**dotenv_values(".env"), **os.environ}
    auth = (env["NROD_EMAIL"], env["NROD_PASSWORD"])
    raw, man = Path("data/raw/nrod"), Path("data/manifests/nrod.csv")
    raw.mkdir(parents=True, exist_ok=True)
    man.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    new = not man.exists()
    with man.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "file", "path", "bytes", "sha256"])
        for typ, name in FILES.items():
            r = requests.get(BASE, params={"type": typ}, auth=auth, timeout=300)
            r.raise_for_status()
            path = raw / f"{today}_{name}"
            path.write_bytes(r.content)
            w.writerow(
                [today, typ, str(path), len(r.content), hashlib.sha256(r.content).hexdigest()]
            )
            print(typ, len(r.content), "bytes")


if __name__ == "__main__":
    main()
