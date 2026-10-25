#!/usr/bin/env python3
"""Extra statistics from the already scored test predictions (no refit, no second scoring):
paired date-bootstrap Brier differences for P(delay >= 15) against historical and persistence,
and the largest absolute errors on held-out routes with their features.
    uv run python scripts/stage3_test_extras.py --run-id RUN
"""

import argparse
import json

import numpy as np

from rail_risk.baselines import models
from rail_risk.network_model import run as R
from rail_risk.network_model import train as T


def paired_brier(frame, p_model, p_base, h, resamples, seed):
    m = frame["horizon_min"].to_numpy() == h
    late = (frame["target_min"].to_numpy()[m] >= models.LATE_MIN).astype(float)
    d = (p_model[m] - late) ** 2 - (p_base[m] - late) ** 2
    dates, idx = np.unique(frame["ssd"].to_numpy()[m], return_inverse=True)
    n, s = np.bincount(idx).astype(float), np.bincount(idx, weights=d)
    counts = np.random.default_rng(seed).multinomial(
        len(dates), np.full(len(dates), 1 / len(dates)), size=resamples
    )
    boot = (counts @ s) / (counts @ n)
    return {
        "diff": float(s.sum() / n.sum()),
        "lo": float(np.quantile(boot, 0.025)),
        "hi": float(np.quantile(boot, 0.975)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    run_id = ap.parse_args().run_id
    cfg = T.load_config()
    rows, held, split, _ = R.prepare_frames(include_test=True)
    hold = rows["test"][rows["test"]["route"].isin(held)]
    base = models.predict(models.fit(rows["train_all_routes"]), hold)
    model = T.Booster3.load(T.MODEL_DIR / run_id)
    mp = model.predict(hold, base["shrunk_persistence"]["point"].to_numpy())
    out = {"run_id": run_id, "brier_paired_model_minus_baseline": {}}
    for name in ("historical", "persistence"):
        out["brier_paired_model_minus_baseline"][name] = {
            h: paired_brier(
                hold, mp["p_late"].to_numpy(), base[name]["p_late"].to_numpy(), h,
                cfg["resamples"], cfg["seed"],
            )
            for h in (15, 60)
        }  # fmt: skip
    err = (mp["point"] - hold["target_min"]).abs()
    top = hold.assign(pred=mp["point"], abs_err=err).sort_values("abs_err", ascending=False).head(8)
    cols = ["ssd", "rid", "route", "horizon_min", "target_min", "pred", "late_now_min",
            "fc_late_min", "n_actuals", "stops_remaining", "sched_min_remaining", "slack_min",
            "dest_cancel_n", "net_late5_share", "abs_err"]  # fmt: skip
    out["largest_errors"] = top[cols].round(2).astype(str).to_dict("records")
    json_path = T.REPORT_DIR / "test_extras.json"
    json_path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
