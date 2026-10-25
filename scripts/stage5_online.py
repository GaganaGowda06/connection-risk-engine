#!/usr/bin/env python3
"""Stage 5 step 4 on VALIDATION: choose the online per-hub step size eta for the chosen option.

Every hub starts from the global calibrated model; an outcome is used only after its outbound has
departed and before the scored row's T (src/rail_risk/connections/online.py). Scored on validation
pairs of held-out service groups, not-yet-arrived rows (the test 5 condition). eta 0 is the static
global model. Rule: lowest pooled Brier with ECE at most 0.05 at every hub, ties within 0.0005
go to the smaller eta.

    uv run python scripts/stage5_online.py --tag options
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from rail_risk.connections import build as B
from rail_risk.connections import data as DD
from rail_risk.connections import online as O
from rail_risk.evaluation import _bin_stats

OUT = Path("reports/connections/stage5")
MODELS = Path("data/interim/stage5/models")


def stats(p, y):
    b, e, _ = _bin_stats(p, y.astype(float), 10)
    return {"n": int(len(y)), "brier": b, "ece": e}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="options")
    ap.add_argument("--option", default=None)
    args = ap.parse_args()
    cfg = B.load_config()
    pick = (
        args.option
        or json.loads((OUT / f"{args.tag}_validation.json").read_text())["summary"]["pick"]
    )
    model = joblib.load(MODELS / f"{args.tag}.joblib")
    va = DD.flag_groups(DD.load_block("validation"), cfg).reset_index(drop=True)
    p = model.predict(va)[f"{pick}_cal"].to_numpy()
    t = O.seconds(va["snap_time"])
    r = O.seconds(va["act_dep_out"])
    y = np.where((va["out_cancel"] == 0) & va["miss"].notna(), va["miss"], np.nan)
    hub = va["hub"].to_numpy()
    score = (va["held_group"] & DD.not_arrived(va) & np.isfinite(y)).to_numpy()
    res = {}
    for eta in O.ETAS:
        q, used = O.run(hub, t, p, y, r, eta)
        assert not (used > t).any()
        cells = {"ALL": stats(q[score], y[score])}
        for h in sorted(set(hub)):
            m = score & (hub == h)
            cells[h] = stats(q[m], y[m])
        for hz in (5, 15, 30, 60):
            m = score & (va["horizon_min"].to_numpy() == hz)
            cells[f"h{hz}"] = stats(q[m], y[m])
        res[str(eta)] = cells
        print(eta, {k: (round(v["brier"], 4), round(v["ece"], 4)) for k, v in cells.items()})
    ok = {
        e: c["ALL"]["brier"]
        for e, c in res.items()
        if all(res[e][h]["ece"] <= 0.05 for h in ("BHM", "MAN", "BRI"))
    }
    best = min(ok.values())
    eta = next(e for e in sorted(ok, key=float) if ok[e] <= best + 0.0005)
    (OUT / f"{args.tag}_online_validation.json").write_text(
        json.dumps({"option": pick, "eta": float(eta), "cells": res}, indent=1)
    )
    print("eta", eta)


if __name__ == "__main__":
    main()
