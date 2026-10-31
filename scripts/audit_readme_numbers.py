#!/usr/bin/env python3
# ruff: noqa: E501
"""Stage 7 audit: every result number in the README appears in the report file it cites.

Each claim is the number as the README writes it and the files that should hold it. A number
matches when some value in a file rounds to it at the README's own precision (JSON values,
markdown and csv cells, and text). Exits 1 when a claim is not found, and lists every number in
the README that no claim covers, so each one is either traced or classed as not a result.

    uv run python scripts/audit_readme_numbers.py
"""

import argparse
import csv
import json
import re
from pathlib import Path

NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
S4 = "reports/network_model/stage4/test_results.json"
S4B = "reports/network_model/stage4b/second_scoring_results.json"
OOT = "reports/serving/oot_table_6a.md"
LOAD = "reports/serving/loadtest_w4/SUMMARY.md"
REC = "reports/serving/recovery_after_sigkill.json"
S5 = "reports/connections/stage5/test_relabel_comparison.md"
S5J = "reports/connections/stage5/test_relabel_results.json"
CL = "reports/capture_loss"
WEEK = [Path(CL) / f"2026-10-{d}.json" for d in range(17, 24)]  # the Darwin week the README cites

# number as written in the README -> files that hold it
CLAIMS = {
    "2.22": [S4], "3.67": [S4], "2.41": [S4], "3.82": [S4], "4.03": [S4],
    "0.803": [S4B], "0.788": [S4B], "0.817": [S4B], "0.791": [S4B], "0.58": [S4B, S4],
    "0.021": [S4B], "0.040": [S4B], "0.051": [S4B], "0.013": [S4B], "0.012": [S4B],
    "0.029": [S5J], "0.801": [S4B], "0.797": [S4B],
    "0.043": ["reports/connections/stage5/options_validation.json"],
    "0.059": ["reports/connections/stage5/options_validation.json"],
    "54": [CL], "80": [CL], "0.05": ["GOAL.md"], "95": ["GOAL.md"],
    "51": [LOAD], "200": [LOAD], "400": [LOAD], "1.3": [REC],
    "0.44": ["reports/network_model/stage4b/adaptive_break_it.json"],
    "-0.016": [S4], "-0.094": [S4], "0.058": [S4],
    "0.0861": ["reports/serving/oot_gap_second_look.json"],
    "0.0379": ["reports/serving/oot_gap_second_look.json"],
    "0.035": [OOT, "reports/serving/oot_results.json"],
}  # fmt: skip
# README numbers that are not results: dates, run ids, versions, counts read from config, examples
NOT_RESULTS = (
    "11 13 1 2 3 4 5 6 7 8 9 15 17 18 20 24 25 10 16 23 26 30 45 60 120 12 0 100 000 "
    "2025 2026 2027 1e197857 c0a08ab4 cd4ac6e9 905346cb 8000 8100 8501 8011 17 90 00 01 08 09 19 49 2.5 55 503 "
)


def values(path: Path) -> list[float]:
    text = path.read_text(errors="ignore")
    out = [float(x) for x in NUM.findall(text)]
    if path.suffix == ".json":
        out += [float(x) for x in re.findall(r"-?\d+\.\d+(?:[eE]-?\d+)?", text)]
    return out


def matches(token: str, files: list[str]) -> list[str]:
    dec = len(token.split(".")[1]) if "." in token else 0
    x, half = float(token), 0.5 * 10**-dec + 1e-9
    hit = []
    for f in files:
        p = Path(f)
        if f == CL:
            vals = [v for q in WEEK for v in values(q)]
        elif p.exists():
            vals = values(p)
        else:
            continue
        near = [v for v in vals if abs(v - x) <= half]
        # percentages in the capture-loss files are written as fractions or percent
        if near or (dec == 0 and any(abs(v * 100 - x) <= 0.5 for v in vals)):
            hit.append(f)
    return hit


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    bad = []
    for tok, files in CLAIMS.items():
        hit = matches(tok, files)
        print(f"{'ok  ' if hit else 'MISS'} {tok:>8}  {hit[0] if hit else files}")
        if not hit:
            bad.append(tok)
    rows = [r for r in Path(OOT).read_text().splitlines() if re.match(r"\| test[234] ", r)]
    within = sum(r.rstrip().endswith("| within |") for r in rows)
    better = sum("outside (better)" in r for r in rows)
    ok = len(rows) == 13 and within == 11 and better == 2
    print(
        f"{'ok  ' if ok else 'MISS'} 11 of 13: {within} within, {better} outside (better) of {len(rows)} rows in {OOT}"
    )
    if not ok:
        bad.append("11 of 13")
    import pandas as pd

    cov = pd.read_csv("reports/stage1_coverage.csv")
    got = (cov.route.nunique(), cov.operator.nunique(), cov.hub.nunique())
    print(f"{'ok  ' if got == (25, 11, 5) else 'MISS'} 25 routes, 11 operators, 5 hubs: {got}")
    if got != (25, 11, 5):
        bad.append("breadth")
    size = Path("data/sample/state_oot/state.sqlite").stat().st_size / 1e6
    print(f"{'ok  ' if abs(size - 2.5) < 0.05 else 'MISS'} 2.5 MB: sample ledger is {size:.2f} MB")
    if abs(size - 2.5) >= 0.05:
        bad.append("2.5")
    readme = Path("README.md").read_text()
    body = re.sub(r"```.*?```", "", readme, flags=re.S)
    left = sorted(
        {t for t in NUM.findall(body) if t not in CLAIMS and t not in NOT_RESULTS.split()}
    )
    print("\nREADME numbers with no claim:", left)
    _ = csv, json
    print(f"\n{len(CLAIMS) - len(bad)} of {len(CLAIMS)} claims traced; misses: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
