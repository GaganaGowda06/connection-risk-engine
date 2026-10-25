#!/usr/bin/env python3
"""Fetch the RDG timetable ZIP, archive it, record the manifest and write data/ref/mct.csv.

    uv run python scripts/fetch_dtd.py [--zip data/raw/dtd/RJTTFnnn.ZIP]   (skip the download)

Credentials are read from .env (NRDP_EMAIL, NRDP_PASSWORD) and never printed.
"""

import argparse
import sys
from pathlib import Path

import httpx

from rail_risk.ingest import dtd, mct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, help="use an already downloaded ZIP")
    args = ap.parse_args()
    headers: dict = {}
    try:
        if args.zip:
            path = args.zip
        else:
            email, password = dtd.credentials()
            with httpx.Client(timeout=300) as client:
                token = dtd.authenticate(client, email, password)
                path, headers = dtd.download(client, token)
        members = dtd.read_members(path)
        head = mct.parse_msn_header(members[".MSN"])
        table = mct.build_mct(members[".MSN"], members[".TSI"], path.name)
    except (dtd.DtdError, mct.MctError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    row = dtd.DtdFile(
        filename=path.name,
        sequence=head["sequence"],
        last_modified=dtd.last_modified_iso(headers),
        msn_generated=head["generated_date"],
        sha256=dtd.sha256(path),
        bytes=path.stat().st_size,
        fetched_at=dtd.now_iso(),
    )
    seen = dtd.MANIFEST.exists() and row.sha256 in dtd.MANIFEST.read_text()
    if not seen:
        dtd.append_manifest(row)
    mct.write_mct(table)
    print(f"{row.filename} seq {row.sequence} generated {row.msn_generated} sha256 {row.sha256}")
    print(f"mct.csv rows: {len(table)} (manifest {'unchanged' if seen else 'appended'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
