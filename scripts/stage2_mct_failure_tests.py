#!/usr/bin/env python3
"""Break the MCT path: a station missing from MSN, a TOC pair missing from TSI, a corrupt or
truncated ZIP, a header without FILE-SPEC=05, a bad token and a bad password. Runs against the
real downloaded feed and the real NRDP endpoint (no download), in a scratch directory, and writes
reports/stage2_mct_failure_tests.json. Credentials are never printed.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import pandas as pd

from rail_risk.baselines import gap_bucket
from rail_risk.ingest import dtd, mct


def attempt(fn) -> dict:
    try:
        return {"result": "returned", "value": fn()}
    except (dtd.DtdError, mct.MctError) as exc:
        return {"result": "raised", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    zip_path = sorted(dtd.RAW_DIR.glob("RJTTF*.ZIP"))[-1]
    members = dtd.read_members(zip_path)
    table = mct.build_mct(members[".MSN"], members[".TSI"], zip_path.name)
    tsi = mct.parse_tsi(members[".TSI"])
    out: dict = {"zip": zip_path.name}

    out["station_missing_from_msn"] = {
        "lookup_ZZZ": mct.mct_lookup(table, "ZZZ", "XC", "VT"),
        "pairs_dropped": int(
            len(pd.DataFrame({"hub": ["ZZZ"], "mct_min": [float("nan")]}))
            - len(
                gap_bucket.label_pairs(
                    pd.DataFrame(
                        {
                            "hub": ["ZZZ"],
                            "mct_min": [float("nan")],
                            "plan_arr_in": [pd.Timestamp("2026-10-14", tz="UTC")],
                            "plan_dep_out": [pd.Timestamp("2026-10-14 00:20", tz="UTC")],
                            "act_arr_in": [pd.Timestamp("2026-10-14", tz="UTC")],
                            "act_dep_out": [pd.Timestamp("2026-10-14 00:20", tz="UTC")],
                        }
                    )
                )
            )
        ),
    }
    row = tsi.iloc[0]
    reverse_has_row = bool(
        ((tsi["crs"] == row.crs) & (tsi["arr_toc"] == "ZZ") & (tsi["dep_toc"] == row.arr_toc)).any()
    )
    out["toc_pair_missing_from_tsi"] = {
        "tsi_row": f"{row.crs} {row.arr_toc}->{row.dep_toc} = {row.minutes:.0f}",
        "lookup_with_tsi_row": mct.mct_lookup(table, row.crs, row.arr_toc, row.dep_toc),
        "lookup_pair_not_in_tsi_falls_back_to_msn": mct.mct_lookup(table, row.crs, "ZZ", "ZY"),
        "msn_value": mct.mct_lookup(table, row.crs, "*", "*"),
        "unlisted_reverse_pair_in_tsi": reverse_has_row,
        "hub_pairs_with_tsi_rows": int(tsi["crs"].isin(["BHM", "MAN", "BRI"]).sum()),
    }

    with tempfile.TemporaryDirectory() as tmp:
        data = zip_path.read_bytes()
        cases = {
            "truncated_zip": data[: len(data) // 2],
            "garbage_file": b"<html>503 service unavailable</html>",
            "empty_file": b"",
        }
        flipped = bytearray(data)
        for i in range(len(flipped) // 2, len(flipped) // 2 + 4000):
            flipped[i] ^= 0xFF
        cases["bit_flips_mid_file"] = bytes(flipped)
        zres = {}
        for name, blob in cases.items():
            p = Path(tmp) / f"{name}.ZIP"
            p.write_bytes(blob)
            zres[name] = attempt(lambda p=p: sorted(dtd.read_members(p)))
        out["corrupt_zip"] = zres
        before = Path(mct.MCT_FILE).read_bytes()
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "scripts/fetch_dtd.py", "--zip", str(Path(tmp) / "truncated_zip.ZIP")],
            capture_output=True,
            text=True,
            check=False,
        )
        out["fetch_script_on_truncated_zip"] = {
            "exit_code": proc.returncode,
            "stderr": proc.stderr.strip()[:200],
            "mct_csv_unchanged": Path(mct.MCT_FILE).read_bytes() == before,
        }

    no_spec = members[".MSN"].replace(b"FILE-SPEC=05", b"FILE-SPEC=04")
    out["header_without_file_spec_05"] = attempt(lambda: len(mct.parse_msn(no_spec)))

    with httpx.Client(timeout=60) as client:
        with tempfile.TemporaryDirectory() as tmp:
            out["expired_or_bad_token"] = attempt(
                lambda: dtd.download(client, "expired-token", Path(tmp))
            )
        out["bad_password"] = attempt(
            lambda: dtd.authenticate(client, "nobody@example.invalid", "wrong-password")
        )
    Path("reports/stage2_mct_failure_tests.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
