#!/usr/bin/env python3
"""Break the Stage 3 model on validation rows (the test block stays single-use):
  1. drop the 'what is ahead' features and retrain with the tuned parameters
  2. shuffle route, rid and operator labels and compare predictions
  3. a held-out route with an hour of messages missing, and rows with no trains ahead
  4. feature importance (gain)
    uv run python scripts/stage3_break_it.py --run-id RUN
Writes reports/network_model/break_it.json.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from rail_risk.baselines import models
from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F
from rail_risk.network_model import run as R
from rail_risk.network_model import train as T


def mae(pred: np.ndarray, df: pd.DataFrame, h: int | None = None) -> float:
    m = np.ones(len(df), bool) if h is None else (df["horizon_min"].to_numpy() == h)
    return float(np.abs(pred[m] - df["target_min"].to_numpy()[m]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    run_id = ap.parse_args().run_id
    cfg = T.load_config()
    rows, held, split, _ = R.prepare_frames(include_test=False)
    train, val = rows["train"], rows["validation"]
    hv = val[val["route"].isin(held)]
    base_fit = models.fit(rows["train_all_routes"])
    base_hv = models.predict(base_fit, hv)["shrunk_persistence"]["point"].to_numpy()
    model = T.Booster3.load(T.MODEL_DIR / run_id)
    full = model.predict(hv, base_hv)["point"].to_numpy()
    out: dict = {"run_id": run_id, "held_out_validation_rows": len(hv)}
    out["full_model_mae"] = {h: round(mae(full, hv, h), 3) for h in (15, 60)}
    out["shrunk_persistence_mae"] = {h: round(mae(base_hv, hv, h), 3) for h in (15, 60)}

    # 1. drop the 'what is ahead' family (same tuned parameters and rounds)
    tuned = json.loads((T.REPORT_DIR / "tuning.json").read_text())
    keep = [c for c in F.FEATURES if c not in F.AHEAD]
    base_tr = models.predict(base_fit, train)["shrunk_persistence"]["point"].to_numpy()
    p = T._params(
        cfg, tuned["median"]["params"], {"objective": "quantile", "alpha": 0.5}, "residual", keep
    )
    b = lgb.train(
        p,
        lgb.Dataset(D.matrix(train, keep), train["target_min"].to_numpy() - base_tr),
        tuned["median"]["rounds"],
    )
    no_ahead = base_hv + b.predict(D.matrix(hv, keep))
    out["without_ahead_features_mae"] = {h: round(mae(no_ahead, hv, h), 3) for h in (15, 60)}
    out["ahead_features_dropped"] = F.AHEAD

    # 2. shuffle the identity labels the model must not use
    rng = np.random.default_rng(5)
    sh = hv.copy()
    for c in ("route", "rid", "toc", "dest_tpl", "from_tpl"):
        sh[c] = rng.permutation(sh[c].to_numpy())
    again = model.predict(sh, base_hv)["point"].to_numpy()
    out["shuffled_labels_max_abs_prediction_change"] = float(np.abs(again - full).max())

    # 3a. an hour of messages missing: rebuild one validation day with 10:00-11:00 UTC removed
    day = sorted(hv["ssd"].unique())[3]
    snaps = pd.read_parquet(Path(f"data/interim/stage2/snapshots/service_date={day}/part.parquet"))
    gap = (pd.Timestamp(f"{day} 10:00", tz="UTC"), pd.Timestamp(f"{day} 11:00", tz="UTC"))
    holidays = F.bank_holidays()
    feats = F.build_day(D.PPORT, day, snaps, holidays, gap=gap)
    fastest = pd.read_parquet(D.FASTEST)
    cut = D.prepare(
        F.add_slack(feats, fastest).merge(
            snaps[[*F.KEY, "target_late_s", "has_target", "toc"]], on=F.KEY, how="left"
        ),
        fastest,
    )
    cut = cut[cut["has_target"] & cut["route"].isin(held)]
    ref = hv[hv["ssd"] == day].merge(cut[F.KEY], on=F.KEY)
    cut = cut.merge(ref[F.KEY], on=F.KEY)

    def after(d: pd.DataFrame) -> pd.Series:
        return d["snap_time"].between(gap[1], gap[1] + pd.Timedelta(hours=1))

    def pr(d: pd.DataFrame) -> np.ndarray:
        base = models.predict(base_fit, d)["shrunk_persistence"]["point"].to_numpy()
        return model.predict(d, base)["point"].to_numpy()

    ref_p, cut_p = pr(ref), pr(cut)
    sel = after(ref).to_numpy()
    out["missing_hour"] = {
        "day": str(day),
        "window_utc": "10:00-11:00",
        "held_out_rows_after_gap": int(sel.sum()),
        "all_predictions_finite": bool(np.isfinite(cut_p).all()),
        "mae_after_gap_normal": round(mae(ref_p[sel], ref[sel]), 3) if sel.any() else None,
        "mae_after_gap_missing": round(mae(cut_p[sel], cut[sel]), 3) if sel.any() else None,
        "rows_with_no_actual_normal": int((ref[sel]["n_actuals"] == 0).sum()),
        "rows_with_no_actual_missing": int((cut[sel]["n_actuals"] == 0).sum()),
    }

    # 3b. no trains ahead: rows where nothing was seen ahead, at the destination or next stop
    none = ((hv["ahead_n"] == 0) & (hv["dest_n"] == 0) & (hv["next_n"] == 0)).to_numpy()
    zeroed = hv.copy()
    for c in F.AHEAD:
        zeroed[c] = 0.0 if c.endswith("_n") else np.nan
    z = model.predict(zeroed, base_hv)["point"].to_numpy()
    out["no_trains_ahead"] = {
        "rows_with_nothing_ahead": int(none.sum()),
        "mae_those_rows_model": round(mae(full[none], hv[none]), 3) if none.any() else None,
        "mae_those_rows_shrunk": round(mae(base_hv[none], hv[none]), 3) if none.any() else None,
        "all_ahead_features_zeroed_mae": {h: round(mae(z, hv, h), 3) for h in (15, 60)},
        "finite": bool(np.isfinite(z).all()),
    }
    # 4. importance
    imp = pd.Series(model.quantiles[0.5].feature_importance("gain"), index=model.features)
    out["importance_gain_top15"] = (
        (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()
    )
    Path("reports/network_model/break_it.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
