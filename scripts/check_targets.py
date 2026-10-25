#!/usr/bin/env python3
"""Run the target sanity check on the train, validation and test blocks.

    uv run python scripts/check_targets.py        (exits 1 on any failure)
Writes reports/target_check.json."""

import json
import sys
from pathlib import Path

from rail_risk.baselines.split import network_split
from rail_risk.baselines.target_check import check_targets
from rail_risk.network_model import dataset as D


def main() -> int:
    split, _ = network_split()
    blocks = {}
    for name, days in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        blocks[name] = D.load(list(days))
    report = check_targets(None, blocks)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/target_check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
