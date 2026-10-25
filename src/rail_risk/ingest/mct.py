"""Minimum connection times from the RDG timetable feed: the MSN and TSI files.

Layout (RSPS5046, fixed width, 82 characters, cols 1-based; links in DECISIONS.md, spec not stored):
  MSN station record: 'A' in col 1; TIPLOC 37-43; CRS 50-52; Minimum Change Time 64-65 (0-99).
  The header is also an 'A' record with FILE-SPEC=05 in cols 31-42 and is skipped.
  A CRS can appear on several 'A' records (one per TIPLOC).
  TSI record: CRS,arriving TOC,departing TOC,minutes[,comment]. It overrides the MSN time for that
  arriving to departing TOC pair only: not bi-directional, all day, every day.
Rule: mct(crs, arr_toc, dep_toc) = TSI if present, else MSN, else None. A station with no MCT, or
with conflicting MCTs on its records, gives None: no default is ever assumed.
"""

import re
from pathlib import Path

import pandas as pd

MCT_FILE = Path("data/ref/mct.csv")
COLUMNS = ["crs", "arr_toc", "dep_toc", "minutes", "source_file", "source_seq", "generated_date"]
ANY = "*"
FILE_SPEC = "FILE-SPEC=05"
_TOC = re.compile(r"[A-Z0-9]{2}")
_CRS = re.compile(r"[A-Z]{3}")


class MctError(RuntimeError):
    """The feed is not in the layout this parser was written for."""


def _lines(raw: bytes) -> list[str]:
    return [ln.rstrip("\r") for ln in raw.decode("latin-1").split("\n")]


def parse_msn_header(raw: bytes) -> dict[str, str]:
    """Sequence and generated date from the '/!!' comments, and the FILE-SPEC=05 check."""
    lines = _lines(raw)
    seq = gen = ""
    spec = False
    for ln in lines:
        if ln.startswith("/!! Sequence:"):
            seq = ln.split(":", 1)[1].strip()
        elif ln.startswith("/!! Generated:"):
            gen = ln.split(":", 1)[1].strip()
        elif ln[:1] == "A" and ln[30:42] == FILE_SPEC:
            spec = True
    if not spec:
        raise MctError(f"MSN header with {FILE_SPEC} in cols 31-42 not found")
    if not re.fullmatch(r"\d{2}/\d{2}/\d{4}", gen):
        raise MctError(f"MSN Generated date not found or unreadable: {gen!r}")
    day, month, year = gen.split("/")
    return {"sequence": seq, "generated_date": f"{year}-{month}-{day}"}


def parse_msn(raw: bytes) -> pd.DataFrame:
    """One row per MSN station record: crs, tiploc, name, minutes (NaN when blank)."""
    parse_msn_header(raw)
    rows = []
    for ln in _lines(raw):
        if ln[:1] != "A" or ln[30:42] == FILE_SPEC:
            continue
        if len(ln) < 65:
            raise MctError(f"short MSN station record ({len(ln)} characters): {ln[:40]!r}")
        crs = ln[49:52]
        if not _CRS.fullmatch(crs):
            raise MctError(f"MSN record without a CRS code in cols 50-52: {ln[:60]!r}")
        field = ln[63:65].strip()
        if field and not field.isdigit():
            raise MctError(f"MSN minimum change time not numeric for {crs}: {field!r}")
        rows.append(
            {
                "crs": crs,
                "tiploc": ln[36:43].strip(),
                "name": ln[5:35].strip(),
                "minutes": float(field) if field else float("nan"),
            }
        )
    if not rows:
        raise MctError("no MSN station records found")
    return pd.DataFrame(rows)


def parse_tsi(raw: bytes) -> pd.DataFrame:
    """One row per TSI record: crs, arr_toc, dep_toc, minutes, comment."""
    rows = []
    for n, ln in enumerate(_lines(raw), start=1):
        if not ln.strip():
            continue
        parts = ln.split(",", 4)
        if len(parts) < 4:
            raise MctError(f"TSI line {n} has fewer than 4 fields: {ln[:60]!r}")
        crs, arr, dep, mins = (p.strip() for p in parts[:4])
        if not (_CRS.fullmatch(crs) and _TOC.fullmatch(arr) and _TOC.fullmatch(dep)):
            raise MctError(f"TSI line {n} has a bad CRS or TOC code: {ln[:60]!r}")
        if not mins.isdigit():
            raise MctError(f"TSI line {n} minutes not numeric: {mins!r}")
        rows.append(
            {
                "crs": crs,
                "arr_toc": arr,
                "dep_toc": dep,
                "minutes": float(mins),
                "comment": parts[4].strip() if len(parts) > 4 else "",
            }
        )
    return pd.DataFrame(rows, columns=["crs", "arr_toc", "dep_toc", "minutes", "comment"])


def build_mct(msn: bytes, tsi: bytes, source_file: str) -> pd.DataFrame:
    """The reference table. MSN rows have arr_toc = dep_toc = '*'. A CRS whose records disagree
    (or are all blank) gets minutes = NaN, so a lookup fails closed instead of choosing."""
    head = parse_msn_header(msn)
    st = parse_msn(msn)
    distinct = {crs: g["minutes"].dropna().unique() for crs, g in st.groupby("crs")}
    msn_rows = pd.DataFrame(
        {
            "crs": list(distinct),
            "arr_toc": ANY,
            "dep_toc": ANY,
            "minutes": [float(v[0]) if len(v) == 1 else float("nan") for v in distinct.values()],
        }
    )
    tsi_rows = parse_tsi(tsi).drop(columns="comment").drop_duplicates(["crs", "arr_toc", "dep_toc"])
    out = pd.concat([msn_rows, tsi_rows], ignore_index=True)
    out["source_file"] = source_file
    out["source_seq"] = head["sequence"]
    out["generated_date"] = head["generated_date"]
    return out[COLUMNS].sort_values(["crs", "arr_toc", "dep_toc"]).reset_index(drop=True)


def write_mct(table: pd.DataFrame, path: Path = MCT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def load_mct_table(path: Path = MCT_FILE) -> pd.DataFrame:
    if not path.exists():
        raise MctError(f"{path} not found: run scripts/fetch_dtd.py (no MCT value is assumed)")
    df = pd.read_csv(path, dtype={"arr_toc": str, "dep_toc": str})
    missing = [c for c in COLUMNS if c not in df]
    if missing:
        raise MctError(f"{path} lacks columns {missing}")
    return df


def mct_lookup(table: pd.DataFrame, crs: str, arr_toc: str, dep_toc: str) -> float | None:
    """TSI for the arriving to departing TOC pair if present, else the MSN time, else None."""
    t = table[table["crs"] == crs]
    pair = t[(t["arr_toc"] == arr_toc) & (t["dep_toc"] == dep_toc)]
    if len(pair):
        v = pair["minutes"].iloc[0]
        return None if pd.isna(v) else float(v)
    base = t[(t["arr_toc"] == ANY) & (t["dep_toc"] == ANY)]
    if base.empty or pd.isna(base["minutes"].iloc[0]):
        return None
    return float(base["minutes"].iloc[0])
