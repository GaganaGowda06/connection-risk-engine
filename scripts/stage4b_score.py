#!/usr/bin/env python3
# ruff: noqa: E501  (analysis script with wide report dictionaries)
"""SECOND SCORING of the test block, with the adaptive method fixed on validation, and February
through the same frozen procedure (a separate line). The first test scoring (static CQR) is in
reports/network_model/stage4/. Each block refuses to run twice (markers).

    uv run python scripts/stage4b_score.py --run-id MODELRUN --mlflow-run-id CALRUN
Primary variant: every route starts the block with no history (state only from the block itself).
Secondary variant (test only): state carried over from the validation block's outcomes."""

import argparse
import json
from datetime import date

import mlflow
import numpy as np
import pandas as pd

from rail_risk.baselines import models, scoring
from rail_risk.network_model import dataset as D
from rail_risk.network_model import stage4b as B
from rail_risk.network_model.evaluate import paired_brier

TEST_MARKER = B.OUT / "test_second_scoring.json"
FEB_MARKER = B.OUT / "february_adaptive.json"
HEADLINE = (15, 60)


def coverage_ci(ssd: np.ndarray, ok: np.ndarray, seed: int = 20261015, n: int = 1000) -> list:
    dates, idx = np.unique(ssd, return_inverse=True)
    cnt, hit = np.bincount(idx).astype(float), np.bincount(idx, weights=ok)
    counts = np.random.default_rng(seed).multinomial(
        len(dates), np.full(len(dates), 1 / len(dates)), size=n
    )
    boot = (counts @ hit) / (counts @ cnt)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def block_report(df: pd.DataFrame, adapted: pd.DataFrame) -> dict:
    y = df["target_min"].to_numpy()
    lo, hi = adapted["lo"].to_numpy(), adapted["hi"].to_numpy()
    s_lo, s_hi = df["static_lo"].to_numpy(), df["static_hi"].to_numpy()
    out: dict = {"rows": len(df)}
    for name, (a, b) in {"static": (s_lo, s_hi), "adaptive": (lo, hi)}.items():
        ok = ((a <= y) & (y <= b)).astype(float)
        out[name] = {"overall": float(ok.mean()), "mean_width": float((b - a).mean())}
        out[name]["overall_ci"] = coverage_ci(df["ssd"].to_numpy(), ok)
        for h in HEADLINE:
            m = df["horizon_min"].to_numpy() == h
            per = pd.Series(ok[m]).groupby(df["route"].to_numpy()[m]).mean()
            out[name][str(h)] = {
                "coverage": float(ok[m].mean()), "ci": coverage_ci(df["ssd"].to_numpy()[m], ok[m]),
                "mean_width": float((b - a)[m].mean()),
                "worst_route": per.idxmin(), "worst_route_coverage": float(per.min()),
            }  # fmt: skip
        out[name]["by_horizon"] = {
            int(h): float(ok[df["horizon_min"].to_numpy() == h].mean())
            for h in sorted(df["horizon_min"].unique())
        }
    out["per_route"] = B.route_table(df, lo, hi, s_lo, s_hi)
    return out


def p_late_report(df: pd.DataFrame, held_all: pd.DataFrame, fit_cold, fit_hist) -> dict:
    """Brier and ECE of the (unchanged) calibrated P(delay >= 15) against the baselines."""
    frame = df.assign(schedule_known=True)
    methods = {"model": df[["point", "p_late"]].assign(lo=df["static_lo"], hi=df["static_hi"])}
    for bar, fit in (("history", fit_hist), ("cold", fit_cold)):
        methods[f"historical_{bar}"] = models.predict(fit, df)["historical"]
    stats = {k: scoring.row_stats(frame, v) for k, v in methods.items()}
    tab = scoring.score_all(
        stats, set(df["route"]), sorted(df["route"].unique()), 1000, 20261015, 0.95
    )
    out: dict = {}
    for k in methods:
        for h in HEADLINE:
            r = tab[(tab.method == k) & (tab.group == "held_out") & (tab.horizon_min == h)].iloc[0]
            out[f"{k}_{h}"] = {
                c: float(r[c]) for c in ("brier", "brier_lo", "brier_hi", "ece", "ece_lo", "ece_hi")
            }
    out["brier_paired_vs_historical_history"] = {
        h: paired_brier(df, df["p_late"].to_numpy(), methods["historical_history"]["p_late"].to_numpy(), h, 1000, 20261015)
        for h in HEADLINE
    }  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--mlflow-run-id", required=True)
    args = ap.parse_args()
    for m in (TEST_MARKER, FEB_MARKER):
        if m.exists():
            raise SystemExit(f"already scored: {m}")
    ch = B.load_choice()["choice"]
    method, gamma = ch["method"], ch["gamma"] or 0.0
    cal, rows, held, fit, gs = B.load_context(args.run_id)
    hist_fit = models.fit(rows["train_all_routes"])
    val, test = rows["validation"], rows["test"]
    warm = B.with_predictions(cal, fit, val[val["route"].isin(held)].reset_index(drop=True))
    hold = B.with_predictions(cal, fit, test[test["route"].isin(held)].reset_index(drop=True))
    out: dict = {"SECOND_SCORING_OF_THE_TEST_BLOCK": True, "run_id": args.run_id,
                 "method": method, "gamma": gamma}  # fmt: skip
    a_cold = B.adaptive_block(hold, gs, method, gamma)
    a_warm = B.adaptive_block(hold, gs, method, gamma, warm=warm)
    out["test_primary_no_history"] = block_report(hold, a_cold)
    out["test_state_from_validation"] = block_report(hold, a_warm)
    out["p_late_unchanged"] = p_late_report(hold, hold, fit, hist_fit)
    hold.assign(adaptive_lo=a_cold["lo"], adaptive_hi=a_cold["hi"], alpha=a_cold["alpha"])[
        ["ssd", "rid", "route", "horizon_min", "target_min", "point", "static_lo", "static_hi",
         "adaptive_lo", "adaptive_hi", "alpha", "snap_time", "pt_arr"]
    ].to_parquet(B.CAL_DIR / args.run_id / "test_adaptive_predictions.parquet", index=False)  # fmt: skip
    # February, same frozen procedure, no history at the start of the block
    fastest = pd.read_parquet(D.FASTEST_COLD)
    days = [date(2026, 2, d) for d in range(1, 29)]
    feb = D.prepare(D.load(days, B.FEB / "features"), fastest)
    feb = D.add_duration(feb, B.FEB / "snapshots")
    feb = feb[feb["has_target"] & feb["route"].isin(held)].reset_index(drop=True)
    feb = B.with_predictions(cal, fit, feb)
    f_ad = B.adaptive_block(feb, gs, method, gamma)
    out["february_no_history"] = block_report(feb, f_ad)
    out["february_p_late_unchanged"] = p_late_report(feb, feb, fit, hist_fit)
    (B.OUT / "second_scoring_results.json").write_text(
        json.dumps(out, indent=1, default=str) + "\n"
    )
    from rail_risk.network_model import run as R

    head, dirty = R.git_state()
    marker = {"run_id": args.run_id, "git_head": head, "dirty": dirty, "second_scoring": True}
    TEST_MARKER.write_text(json.dumps(marker) + "\n")
    FEB_MARKER.write_text(json.dumps(marker) + "\n")
    R._tracking()
    with mlflow.start_run(run_id=args.mlflow_run_id):
        mlflow.set_tags({"second_test_scoring_at_head": f"{head}{'+dirty' if dirty else ''}",
                         "adaptive_method": f"{method}:{gamma}"})  # fmt: skip
        for k, blk in (
            ("test", out["test_primary_no_history"]),
            ("feb", out["february_no_history"]),
        ):
            mlflow.log_metric(f"adaptive_{k}_coverage", blk["adaptive"]["overall"])
            mlflow.log_metric(f"adaptive_{k}_width", blk["adaptive"]["mean_width"])
    for k in ("test_primary_no_history", "test_state_from_validation", "february_no_history"):
        b = out[k]
        print(k, "static", round(b["static"]["overall"], 3), "adaptive", round(b["adaptive"]["overall"], 3),
              {h: (round(b["adaptive"][h]["coverage"], 3), b["adaptive"][h]["worst_route"], round(b["adaptive"][h]["worst_route_coverage"], 3)) for h in ("15", "60")})  # fmt: skip


if __name__ == "__main__":
    main()
