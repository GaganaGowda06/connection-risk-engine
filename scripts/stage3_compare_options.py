#!/usr/bin/env python3
"""Compare three model options on a small sample of train days, scored on validation only:
  residual  LightGBM on lateness minus shrunk persistence (the Stage 2 proposal)
  raw       LightGBM on lateness (LightGBM rejects monotone constraints with quantile objectives)
  recovery  LightGBM on minutes recovered (late now minus late at arrival), slack features only
All three use the L1 objective, fixed parameters and no early stopping, so validation is only read
for scoring. Writes reports/network_model/option_comparison.json.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from rail_risk.network_model import dataset as D

PARAMS = dict(
    objective="quantile", alpha=0.5, learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, verbose=-1, num_threads=4, seed=1,
)  # fmt: skip
ROUNDS = 300
RECOVERY_FEATURES = [
    "late_now_min", "horizon_min", "n_actuals", "stops_remaining", "sched_min_remaining",
    "fastest_min_remaining", "slack_min", "late_minus_slack_min", "late_chg_1_min",
    "late_chg_3_min", "fc_late_min",
]  # fmt: skip


def main() -> None:
    split, held = D.load_split()
    days = [*split.train, *split.validation]
    fastest = D.ensure_fastest([*split.train, *split.validation, *split.test], list(split.train))
    df = D.prepare(D.load(days), fastest)
    rows = D.split_rows(df, split, held)
    sample_days = set(list(split.train)[::4])
    train = rows["train"][rows["train"]["ssd"].isin(sample_days)]
    val = rows["validation"]
    fit_all = rows["train_all_routes"]
    base_tr = D.base_forecasts(fit_all, train)["shrunk_persistence"]["point"].to_numpy()
    base_va = D.base_forecasts(fit_all, val)["shrunk_persistence"]["point"].to_numpy()
    persistence_va = D.base_forecasts(fit_all, val)["persistence"]["point"].to_numpy()
    y_tr, y_va = train["target_min"].to_numpy(), val["target_min"].to_numpy()
    now_tr = train["late_now_min"].fillna(0.0).to_numpy()
    now_va = val["late_now_min"].fillna(0.0).to_numpy()
    out: dict = {"train_days": len(sample_days), "train_rows": len(train), "val_rows": len(val)}

    xt, xv = D.matrix(train), D.matrix(val)
    preds = {}
    m = lgb.train(PARAMS, lgb.Dataset(xt, y_tr - base_tr), ROUNDS)
    preds["residual"] = base_va + m.predict(xv)
    m = lgb.train(PARAMS, lgb.Dataset(xt, y_tr), ROUNDS)
    preds["raw"] = m.predict(xv)
    rt, rv = D.matrix(train, RECOVERY_FEATURES), D.matrix(val, RECOVERY_FEATURES)
    m = lgb.train(PARAMS, lgb.Dataset(rt, now_tr - y_tr), ROUNDS)
    preds["recovery"] = now_va - m.predict(rv)
    preds["shrunk_persistence"] = base_va
    preds["persistence"] = persistence_va
    held_mask = val["route"].isin(held).to_numpy()
    table = []
    for name, p in preds.items():
        for h in (15, 60):
            for grp, mk in (("held_out", held_mask), ("train_routes", ~held_mask)):
                sel = mk & (val["horizon_min"].to_numpy() == h)
                table.append(
                    {"option": name, "horizon_min": h, "group": grp, "n": int(sel.sum()),
                     "mae": round(float(np.abs(p[sel] - y_va[sel]).mean()), 3)}
                )  # fmt: skip
    out["validation_mae"] = table
    Path("reports/network_model").mkdir(parents=True, exist_ok=True)
    Path("reports/network_model/option_comparison.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print(
        pd.DataFrame(table)
        .pivot_table(index=["group", "horizon_min"], columns="option", values="mae")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
