#!/usr/bin/env python3
"""Stage 4 step 1: retrain and gate on validation in each history mode.

  history       base and slack table use every route's own train-date history (Stage 3)
  cold          held-out routes get no route history: base fitted on the training routes only and
                a slack table built without the held-out services
  warm          history, plus the slot-history features (the route's own previous 28 days)
  warm_drift    warm, and the base is lateness now plus the route's recent median recovery
  cold_noslack  cold, and the slack and slot-history features are dropped (they need route
                history, which a day-one route does not have)
    uv run python scripts/stage4_modes.py [--only cold_noslack]
Writes reports/network_model/stage4/<name>/validation_gate.json and mode_comparison.json.
"""

import argparse
import json
from pathlib import Path

from rail_risk.logging import configure_logging
from rail_risk.network_model import features as F
from rail_risk.network_model import run as R

OUT = Path("reports/network_model/stage4")
MODES = {
    "history": ("history", F.HISTORY, False),
    "warm": ("history", None, False),
    "warm_drift": ("history", None, True),
    "cold": ("cold", F.HISTORY, False),
    "cold_noslack": ("cold", [*F.SLACK, *F.HISTORY], False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(MODES))
    names = ap.parse_args().only or list(MODES)
    configure_logging("INFO", False)
    path = OUT / "mode_comparison.json"
    keep = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        mode, drop, warm = MODES[name]
        v = R.train_command(mode=mode, out_dir=OUT / name, drop=drop, warm_base=warm)
        keep[name] = {k: v[k] for k in ("run_id", "gate", "tuned", "git_head")}
    path.write_text(json.dumps(keep, indent=1, default=str) + "\n")
    for name, v in keep.items():
        print(name, v["run_id"])
        for g in v["gate"]:
            if g["baseline"] in ("persistence", "shrunk_persistence"):
                print(
                    f"  vs {g['baseline']:20s} {g['horizon_min']:>3} model {g['model_mae']:.3f} "
                    f"base {g['baseline_mae']:.3f} diff {g['diff']:.3f} "
                    f"[{g['lo']:.3f}, {g['hi']:.3f}]"
                )


if __name__ == "__main__":
    main()
