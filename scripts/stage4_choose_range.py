#!/usr/bin/env python3
"""Stage 4 step 2: compare three range methods on validation only (held-out routes).

    uv run python scripts/stage4_choose_range.py --run-id RUN
RUN is the cold_noslack model. Writes reports/network_model/stage4/range_methods.json."""

import argparse
import json
from pathlib import Path

import pandas as pd

from rail_risk.baselines import models
from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F
from rail_risk.network_model import stage4 as S
from rail_risk.network_model import train as T

OUT = Path("reports/network_model/stage4")
MODES = {
    "history": (False, F.HISTORY, False),
    "warm": (False, None, False),
    "warm_drift": (False, None, True),
    "cold_noslack": (True, [*F.SLACK, *F.HISTORY], False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--name", default="cold_noslack", choices=list(MODES))
    args = ap.parse_args()
    run_id, name = args.run_id, args.name
    cold, drop, warm = MODES[name]
    cfg = {**T.load_config(), "drop": drop, "warm_base": warm}
    tuned = json.loads((OUT / name / "tuning.json").read_text())
    rows, held, split = S.prepare_calibration_frames(cold=cold)
    train, val = rows["train"], rows["validation"]
    base_rows = train if cold else rows["train_all_routes"]
    model = T.Booster3.load(T.MODEL_DIR / run_id)
    S.POOL_DIR.mkdir(parents=True, exist_ok=True)
    cache = S.POOL_DIR / f"fold_pool_{run_id}.parquet"
    if cache.exists():
        fold = pd.read_parquet(cache)
    else:
        fold = S.fold_pool(train, cfg, tuned, None if cold else base_rows)
        fold.to_parquet(cache, index=False)
    own = S.final_pool_rows(model, base_rows, val[~val["route"].isin(held)], warm)
    pool = pd.concat([fold, own], ignore_index=True)
    hv = val[val["route"].isin(held)].reset_index(drop=True)
    base = D.base_point(models.fit(base_rows), hv, warm)
    pred = model.predict(hv, base)
    res = S.compare_methods(pool, hv, pred)
    out = {
        "name": name,
        "run_id": run_id,
        "pool_rows": len(pool),
        "fold_rows": len(fold),
        "val_rows": len(own),
    }
    out["summary"] = {k: S.brief(v) for k, v in res.items()}
    out["detail"] = res
    (OUT / f"range_methods_{name}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(pd.DataFrame(out["summary"]).T.to_string())
    for m in ("raw", "cross", "cqr", "mondrian"):
        print(m, {k: round(v, 3) for k, v in res[m][15]["routes"].items()})
        print(m, {k: round(v, 3) for k, v in res[m][60]["routes"].items()})


if __name__ == "__main__":
    main()
