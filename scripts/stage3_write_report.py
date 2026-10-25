#!/usr/bin/env python3
"""Write reports/network_model/stage3_report.md from the scored CSVs and JSON files."""

import json
from pathlib import Path

import pandas as pd

R = Path("reports/network_model")
HELD = ["lst-cbg", "myb-bmo", "bhm-liv", "man-shf", "glq-abd", "bri-cdf", "bri-wsm"]


def f(x: float, d: int = 3) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def main() -> None:
    gate = json.loads((R / "validation_gate.json").read_text())
    marker = json.loads((R / "test_scored.json").read_text())
    paired = pd.read_csv(R / "stage3_test_paired.csv")
    score = pd.read_csv(R / "stage3_test_scores.csv")
    extras = json.loads((R / "test_extras.json").read_text())
    brk = json.loads((R / "break_it.json").read_text())
    L = [
        "# Stage 3: route-agnostic model",
        "",
        f"Run {gate['run_id']} (training commit {gate['git_head']}). Option: residual over shrunk "
        "persistence, one LightGBM for all routes, no route, station or operator identity. Test "
        f"block 2026-10-10 to 2026-10-16, held-out routes only ({marker['rows']} rows over six "
        "horizons), scored once. Intervals are 95% date-bootstrap CIs.",
        "",
        "## Test 2: MAE, model minus baseline (paired, minutes)",
        "",
        "| baseline | minutes before | n | model MAE | baseline MAE | difference [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for r in paired.itertuples():
        L.append(
            f"| {r.baseline} | {r.horizon_min} | {r.n} | {f(r.model_mae)} | {f(r.baseline_mae)} | "
            f"{f(r.diff)} [{f(r.lo)}, {f(r.hi)}] |"
        )
    L += [
        "",
        "## Tests 3 and 4 on held-out routes together",
        "",
        "| method | minutes before | MAE | raw 80% coverage | Brier | ECE |",
        "|---|---|---|---|---|---|",
    ]
    for m in ("model", "persistence", "shrunk_persistence", "historical"):
        for h in (15, 60):
            r = score[(score.method == m) & (score.group == "held_out") & (score.horizon_min == h)]
            r = r.iloc[0]
            L.append(
                f"| {m} | {h} | {f(r.mae, 2)} | {f(r.coverage)} [{f(r.coverage_lo)}, "
                f"{f(r.coverage_hi)}] | {f(r.brier)} [{f(r.brier_lo)}, {f(r.brier_hi)}] | "
                f"{f(r.ece)} [{f(r.ece_lo)}, {f(r.ece_hi)}] |"
            )
    L += ["", "Paired Brier difference, model minus baseline, P(delay >= 15):", ""]
    for name, d in extras["brier_paired_model_minus_baseline"].items():
        for h, r in d.items():
            L.append(
                f"- against {name} at {h} minutes: {f(r['diff'], 4)} [{f(r['lo'], 4)}, "
                f"{f(r['hi'], 4)}]"
            )
    L += [
        "",
        "## Per held-out route (model, then shrunk persistence)",
        "",
        "Greater Anglia (lst-cbg) and Chiltern (myb-bmo) never appear in training. bri-wsm still "
        "carries about 2 percent of stops with arrival or departure inversions of two minutes or "
        "more (Stage 1), so its actual times carry extra noise.",
        "",
        "| route | minutes before | n | model MAE | shrunk MAE | model coverage | model Brier "
        "| model ECE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for route in HELD:
        for h in (15, 60):
            a = score[(score.method == "model") & (score.group == route) & (score.horizon_min == h)]
            b = score[
                (score.method == "shrunk_persistence")
                & (score.group == route)
                & (score.horizon_min == h)
            ]
            a, b = a.iloc[0], b.iloc[0]
            L.append(
                f"| {route} | {h} | {int(a.n)} | {f(a.mae, 2)} | {f(b.mae, 2)} | {f(a.coverage)} | "
                f"{f(a.brier)} | {f(a.ece)} |"
            )
    L += ["", "## Break it (validation rows, held-out routes)", ""]
    L.append(
        f"- full model MAE {brk['full_model_mae']}; "
        f"shrunk persistence {brk['shrunk_persistence_mae']}"
    )
    L.append(f"- without the what-is-ahead features: {brk['without_ahead_features_mae']}")
    L.append(
        f"- shuffled route, rid and operator labels, largest prediction change: "
        f"{brk['shuffled_labels_max_abs_prediction_change']}"
    )
    L.append(f"- missing hour: {brk['missing_hour']}")
    L.append(f"- no trains ahead: {brk['no_trains_ahead']}")
    L.append(
        f"- feature importance (share of gain, top 10): "
        f"{dict(list(brk['importance_gain_top15'].items())[:10])}"
    )
    (R / "stage3_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
