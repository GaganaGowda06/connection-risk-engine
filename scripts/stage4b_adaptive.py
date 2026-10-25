#!/usr/bin/env python3
"""Stage 4 finish, step 3: compare three adaptive range methods on validation only, run
sequentially in time order, each route or group starting from the global CQR with no history.

    uv run python scripts/stage4b_adaptive.py --run-id MODELRUN
Writes reports/network_model/stage4b/adaptive_methods.json and adaptive_choice.json."""

import argparse
import json
from pathlib import Path

import pandas as pd

from rail_risk.baselines import models
from rail_risk.network_model import adaptive as A
from rail_risk.network_model import calibrate as C
from rail_risk.network_model import dataset as D
from rail_risk.network_model import stage4 as S

OUT = Path("reports/network_model/stage4b")
CAL_DIR = Path("data/interim/stage4/calibrated")
GAMMAS = [0.002, 0.005, 0.01, 0.02, 0.05]


def global_scores(run_id: str, rows: dict, held: set[str], model) -> A.GlobalScores:
    train, val = rows["train"], rows["validation"]
    fold = pd.read_parquet(S.POOL_DIR / f"fold_pool_{run_id}.parquet")
    own = S.final_pool_rows(model, train, val[~val["route"].isin(held)])
    return A.GlobalScores.from_pool(pd.concat([fold, own], ignore_index=True))


def validation_rows(cal, rows: dict, held: set[str]) -> pd.DataFrame:
    train, val = rows["train"], rows["validation"]
    hv = val[val["route"].isin(held)].reset_index(drop=True)
    pred = cal.predict(hv, D.base_point(models.fit(train), hv))
    return hv.assign(raw_lo=pred["raw_lo"], raw_hi=pred["raw_hi"], point=pred["point"],
                     static_lo=pred["lo"], static_hi=pred["hi"])  # fmt: skip


def pick(table: list[dict]) -> dict:
    """Rule: overall coverage at least 0.01 inside the 0.78 to 0.82 band at 15 and 60 (a margin so
    a fresh block does not fall out of it), then the best worst-route coverage, then the narrower
    mean width. The first version of the rule had no margin; the margin was added after seeing
    validation (on validation only) because the best worst-route setting sat on the band edge."""

    def key(r: dict):
        band = all(0.79 <= r[k] <= 0.81 for k in ("cov15", "cov60"))
        return (band, min(r["worst15_value"], r["worst60_value"]), -r["mean_width"])

    return max((r for r in table if r["method"] != "static"), key=key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    run_id = ap.parse_args().run_id
    OUT.mkdir(parents=True, exist_ok=True)
    cal = C.Calibrated.load(CAL_DIR / run_id)
    rows, held, split = S.prepare_calibration_frames(cold=True)
    gs = global_scores(run_id, rows, held, cal.model)
    hv = validation_rows(cal, rows, held)
    table = [
        {"method": "static", "gamma": None, **A.block_summary(hv, hv["static_lo"], hv["static_hi"])}
    ]
    for method in A.METHODS:
        for gamma in GAMMAS if method != "rolling" else [None]:
            out = A.run(hv, gs, method, gamma or 0.0)
            table.append(
                {"method": method, "gamma": gamma, **A.block_summary(hv, out["lo"], out["hi"])}
            )
    best = pick(table)
    (OUT / "adaptive_methods.json").write_text(json.dumps(table, indent=1, default=str) + "\n")
    (OUT / "adaptive_choice.json").write_text(
        json.dumps({"run_id": run_id, "choice": best, "rule": pick.__doc__}, indent=1, default=str)
        + "\n"
    )
    df = pd.DataFrame(table)
    cols = [
        "method",
        "gamma",
        "overall",
        "cov15",
        "cov60",
        "worst15",
        "worst60",
        "mean_width",
        "first7_overall",
        "first7_worst",
        "last7_overall",
    ]
    pd.set_option("display.width", 250)
    print(df[cols].round(3).to_string(index=False))
    print("CHOICE", best["method"], best["gamma"])


if __name__ == "__main__":
    main()
