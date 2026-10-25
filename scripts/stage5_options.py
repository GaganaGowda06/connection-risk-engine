#!/usr/bin/env python3
"""Stage 5 option comparison on VALIDATION only: physical, LightGBM, logistic, plus the bar.

Scored on validation pairs of held-out service groups (never in a fit or a calibration), split
into not-yet-arrived pairs (the test 5 condition) and all pairs. The test block is not read.
Selection rule, written before the run: lowest pooled Brier of the calibrated probability on
not-yet-arrived held-out-group pairs among options whose ECE is at most 0.05 at every hub; options
within 0.0005 Brier of the best are tied and the simpler one (logit, phys, lgbm) wins.

    uv run python scripts/stage5_options.py [--tag TAG]
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from rail_risk.connections import baseline as BL
from rail_risk.connections import build as B
from rail_risk.connections import data as DD
from rail_risk.connections import pipeline as P
from rail_risk.evaluation import _bin_stats

OUT = Path("reports/connections/stage5")
MODELS = Path("data/interim/stage5/models")
ORDER = ("logit", "phys", "lgbm")


def cell(p, y):
    b, e, _ = _bin_stats(np.asarray(p), np.asarray(y, float), 10)
    return {"n": int(len(y)), "brier": b, "ece": e, "base_rate": float(np.mean(y))}


def score(df, preds, subset):
    y = df["miss"].to_numpy(float)
    out = {}
    for hub in ("ALL", *sorted(df["hub"].unique())):
        for h in ("all", 5, 15, 30, 60):
            m = np.ones(len(df), bool)
            if hub != "ALL":
                m &= (df["hub"] == hub).to_numpy()
            if h != "all":
                m &= (df["horizon_min"] == h).to_numpy()
            if not m.any():
                continue
            out[f"{subset}|{hub}|{h}"] = {k: cell(v[m], y[m]) for k, v in preds.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="options")
    args = ap.parse_args()
    cfg = B.load_config()
    tr = DD.labelled(DD.flag_groups(DD.load_block("train"), cfg))
    va = DD.labelled(DD.flag_groups(DD.load_block("validation"), cfg))
    tr_ok = tr[~tr.held_group].reset_index(drop=True)
    va_ok = va[~va.held_group].reset_index(drop=True)
    va_held = va[va.held_group].reset_index(drop=True)
    assert not tr_ok["held_group"].any() and not va_ok["held_group"].any()
    model = P.fit_all(tr_ok, va_ok)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / f"{args.tag}.joblib")
    bar = BL.fit(tr_ok)
    pred = model.predict(va_held)
    preds = {c: pred[c].to_numpy() for c in pred}
    preds["gap_bucket"] = BL.predict(bar, va_held)
    nya = DD.not_arrived(va_held).to_numpy()
    res = {
        **score(va_held[nya], {k: v[nya] for k, v in preds.items()}, "not_arrived"),
        **score(va_held, preds, "all"),
    }
    # selection
    cand = {}
    for n in P.NAMES:
        hubs = [
            k
            for k in res
            if k.startswith("not_arrived|") and k.endswith("|all") and "|ALL|" not in k
        ]
        ece_ok = all(res[k][f"{n}_cal"]["ece"] <= 0.05 for k in hubs)
        cand[n] = (res["not_arrived|ALL|all"][f"{n}_cal"]["brier"], ece_ok)
    ok = {n: b for n, (b, e) in cand.items() if e}
    best = min(ok.values())
    pick = next(n for n in ORDER if n in ok and ok[n] <= best + 0.0005)
    summary = {
        "pick": pick,
        "candidates": cand,
        "meta": model.meta,
        "rows": {
            "train_eligible": len(tr_ok),
            "val_eligible": len(va_ok),
            "val_held": len(va_held),
            "val_held_not_arrived": int(nya.sum()),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}_validation.json").write_text(
        json.dumps({"summary": summary, "cells": res}, indent=1, default=float)
    )
    print(json.dumps(summary, indent=1, default=float))
    for k in (
        "not_arrived|ALL|all",
        "all|ALL|all",
        *[f"not_arrived|ALL|{h}" for h in (5, 15, 30, 60)],
    ):
        print(k, {n: (round(v["brier"], 4), round(v["ece"], 4)) for n, v in res[k].items()})


if __name__ == "__main__":
    main()
