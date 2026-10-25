#!/usr/bin/env python3
"""Widen the tuning grid once (the Stage 3 best sat on the grid edge) and compare on validation.

    uv run python scripts/stage4_widegrid.py
Writes reports/network_model/stage4/cold_noslack_widegrid/. The chosen model is not changed."""

from pathlib import Path

from rail_risk.logging import configure_logging
from rail_risk.network_model import features as F
from rail_risk.network_model import run as R

GRID = {"num_leaves": [15, 31, 63, 127], "min_data_in_leaf": [25, 50, 100, 200, 400]}


def main() -> None:
    configure_logging("INFO", False)
    out = Path("reports/network_model/stage4/cold_noslack_widegrid")
    v = R.train_command(mode="cold", out_dir=out, grid=GRID, drop=[*F.SLACK, *F.HISTORY])
    print(v["run_id"], v["tuned"])
    for g in v["gate"]:
        if g["baseline"] in ("persistence", "shrunk_persistence"):
            print(g["baseline"], g["horizon_min"], round(g["model_mae"], 3), round(g["diff"], 3))


if __name__ == "__main__":
    main()
