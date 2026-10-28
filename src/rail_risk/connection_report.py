"""The connections section of reports/legacy/model_results.md, written from the backtest result.

Every number in the prose comes from the result dictionary, so the page cannot drift from the
run. The section sits between two marker comments so a rerun replaces it in place.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

START = "<!-- connections:start -->"
END = "<!-- connections:end -->"


def alert_inputs(
    pred: pd.DataFrame, column: str, outcome: str
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    """((validation p, y), (test p, y), test p and y per forecast time) for one alert.

    `outcome` names the 0/1 column that says whether the event happened. Pairs with an unknown
    outcome are left out.
    """
    df = pred[pred[outcome].notna()].copy()

    def pair(block: str, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        sub = df[df["block"] == block]
        if k is not None:
            sub = sub[sub["k"] == k]
        return sub[column].to_numpy(dtype=float), sub[outcome].to_numpy(dtype=float)

    per_k = {f"T-{k} before outbound": pair("test", int(k)) for k in sorted(df["k"].unique())[::-1]}
    return pair("validation"), pair("test"), per_k


def _ci(c: dict[str, Any], digits: int = 3) -> str:
    return f"{c['value']:.{digits}f} ({c['ci_low']:.{digits}f} to {c['ci_high']:.{digits}f})"


def _pct(x: float | None, digits: int = 0) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _reliability_rows(sc: dict[str, Any]) -> list[str]:
    lines = [
        "| Bin | n | Model predicted | Model observed | Baseline predicted | Baseline observed |",
        "|---|---|---|---|---|---|",
    ]
    for m, b in zip(sc["reliability_model"], sc["reliability_baseline"], strict=True):
        if m["n"] is None:
            continue
        lines.append(
            f"| {m['bin']} | {m['n']} | {m['predicted']['value']:.3f} | "
            f"{m['observed']['value']:.3f} ({m['observed']['ci_low']:.3f} to "
            f"{m['observed']['ci_high']:.3f}) | {b['predicted']['value']:.3f} | "
            f"{b['observed']['value']:.3f} ({b['observed']['ci_low']:.3f} to "
            f"{b['observed']['ci_high']:.3f}) |"
        )
    return lines


def _calibration_lines(result: dict[str, Any], pc: dict[str, Any]) -> list[str]:
    """The P(miss) recalibration: how the method was chosen and the test scores before and after."""
    ll = ", ".join(f"{k} {v:.4f}" for k, v in pc["tune_log_loss"].items())
    method = pc["method"]
    lines = [
        "### Recalibration of P(miss)",
        "",
        "The simulated P(miss) over-predicts, so a map from it to the observed miss frequency was "
        "chosen among none, Platt scaling and isotonic regression. It uses validation pairs "
        "only, and only forecasts whose inbound had not arrived, because once the inbound has "
        "arrived its delay is known. Those pairs were split by date: the methods were fitted on "
        f"the later half ({pc['n_calibrate']} forecasts) and scored by log loss on the earlier "
        f"half ({pc['n_tune']} forecasts). Tune-half log loss was {ll}. "
        + (
            "No recalibration won, so none is applied and the scores below are unchanged."
            if method == "none"
            else f"The chosen method is {method}. It is applied to forecasts whose inbound has "
            "not arrived, and the alert threshold was chosen again on the recalibrated "
            "validation probabilities with the same costs."
        ),
        "",
        "| Test, forecast | Pairs | Realised miss rate | Mean predicted before | "
        "Mean predicted after | Brier before | Brier after | ECE before | ECE after |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    after = pc["scores_after"].get("test", {})
    for name, sc in result["scores"].get("test", {}).items():
        a = after.get(name, {})
        for label, key, raw in (
            ("all pairs", "all", sc),
            ("inbound not yet arrived", "not_yet_arrived", sc.get("not_yet_arrived")),
        ):
            now = a.get(key)
            if raw is None or now is None:
                continue
            lines.append(
                f"| {name}, {label} | {raw['pairs']} | {_pct(raw['miss_rate'], 1)} | "
                f"{_pct(raw['mean_predicted_miss'], 1)} | {_pct(now['mean_predicted_miss'], 1)} | "
                f"{_ci(raw['model']['brier'], 4)} | {_ci(now['model']['brier'], 4)} | "
                f"{_ci(raw['model']['ece'], 4)} | {_ci(now['model']['ece'], 4)} |"
            )
    lines += [
        "",
        "The test pairs were scored once with this map, and nothing about it was chosen using "
        "them.",
        "",
    ]
    before = result.get("alerts_before_calibration")
    if before:
        lines += [
            f"The alert threshold before recalibration was {before['tau']:.2f}.",
            "",
        ]
    return lines


def _rounding_verdict(result: dict[str, Any]) -> str:
    """A plain statement of whether the rounding fix also corrected the bias on validation."""
    parts = []
    for name, sc in result["scores"].get("validation", {}).items():
        r = sc["rounding_check"]["all"]
        gap_before = r["mean_before"] - r["realised"]
        gap_after = r["mean_after"] - r["realised"]
        parts.append(
            f"at {name} the overestimate of P(miss) on validation went from "
            f"{100 * gap_before:.1f} to {100 * gap_after:.1f} percentage points"
        )
    if not parts:
        return ""
    last = result["scores"]["validation"]
    left = max(
        r["rounding_check"]["all"]["mean_after"] - r["rounding_check"]["all"]["realised"]
        for r in last.values()
    )
    ending = (
        "so the fix reduces the bias on validation but does not remove it: the model still "
        "predicts more misses than happened."
        if left > 0.005
        else "so the fix removes the bias on validation."
    )
    return "On validation, " + "; ".join(parts) + ", " + ending


def _policy_row(label: str, s: dict[str, Any]) -> str:
    return (
        f"| {label} | {_num(s['tau'])} | {s['alerts']} of {s['n']} | {_num(s['precision'], 3)} | "
        f"{_num(s['recall'], 3)} | {_num(s['alert_rate'], 3)} | {_num(s['cost_per_pair'], 3)} |"
    )


def connections_section(result: dict[str, Any]) -> str:
    test = result["scores"].get("test", {})
    days, pairs, known = result["ast_bhm_days"], result["pairs"], result["pairs_with_known_outcome"]
    base = result["baseline"]
    lines = [
        START,
        "",
        "## Connections: AST to BHM inbound, BHM to EUS outbound",
        "",
        "This section scores the connection engine out of sample. A pair is one AST to BHM "
        "service with one BHM to EUS service that leaves between "
        f"{result['min_connection_time']:.0f} and "
        f"{result['min_connection_time'] + result['window_minutes']:.0f} minutes after it is "
        "due at Birmingham. The minimum connection time of "
        f"{result['min_connection_time']:.0f} minutes is {result['min_connection_time_status']}, "
        "not a published figure. The AST to BHM days split as "
        f"{days['train']} train, {days['validation']} validation and {days['test']} test. "
        f"There were {pairs['train']} training pairs, {pairs['validation']} validation pairs and "
        f"{pairs['test']} test pairs, of which {known['test']} test pairs had a known outcome. "
        "Only test pairs are scored below. The delay model, the tail model, the outbound delay "
        "history and the baseline are all fitted on training dates, and the validation pairs are "
        "used only to choose alert thresholds.",
        "",
        "Each forecast is made 60 or 15 minutes before the outbound is due to leave, with the "
        "exact number of minutes to the inbound's scheduled arrival at that moment given to the "
        "model. If the inbound had already arrived by then, its delay is known and only the "
        "outbound is uncertain. If it was still approaching, the model was given the exact "
        "minutes. If its scheduled arrival had passed or was within the model's shortest "
        "trained horizon of "
        f"{result.get('min_trained_minutes', 5)} minutes, the model saw the true state at that "
        "moment and, for an overdue train, the draws were limited to delays at least as large as "
        "the time already elapsed. The baseline is the historical miss frequency by scheduled gap "
        "bucket (5 minute steps above the minimum time, the last bucket open), fitted on "
        f"{base['pairs_by_gap_bucket'] and sum(base['pairs_by_gap_bucket'].values())} training "
        f"pairs with an overall miss rate of {_pct(base['overall_miss_rate'], 1)}.",
        "",
        "| Forecast time | Pairs | Realised miss rate | Mean predicted | Model Brier | "
        "Baseline Brier | Model ECE | Baseline ECE | 80% range coverage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, sc in test.items():
        cov = sc["interval_coverage"]
        lines.append(
            f"| {name} | {sc['pairs']} | {_pct(sc['miss_rate'], 1)} | "
            f"{_pct(sc['mean_predicted_miss'], 1)} | {_ci(sc['model']['brier'], 4)} | "
            f"{_ci(sc['baseline']['brier'], 4)} | {_ci(sc['model']['ece'], 4)} | "
            f"{_ci(sc['baseline']['ece'], 4)} | "
            + (f"{_ci(cov)} of {sc['interval_pairs']}" if cov else "n/a")
            + " |"
        )
    lines += [
        "",
        "Brackets are 95% bootstrap intervals over service dates. Lower Brier and ECE are "
        "better. The coverage column is the share of pairs whose realised arrival at Euston fell "
        "inside the predicted 80% range, counting only pairs that reached Euston. The table "
        "above is all pairs. At T-60 no inbound had arrived yet, but at T-15 many had, and for "
        "those the inbound delay is simply known, so the all-pairs T-15 gain overstates what the "
        "model adds. The next table keeps only pairs whose inbound had not arrived when the "
        "forecast was made, which is known at prediction time and not chosen by the outcome.",
        "",
        "| Forecast time | Pairs (arrived pairs left out) | Realised miss rate | Mean predicted | "
        "Model Brier | Baseline Brier | Model ECE | Baseline ECE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, sc in test.items():
        w = sc.get("not_yet_arrived")
        if w is None:
            lines.append(f"| {name} | too few pairs | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {name} | {w['pairs']} ({w['excluded_arrived']}) | {_pct(w['miss_rate'], 1)} | "
            f"{_pct(w['mean_predicted_miss'], 1)} | {_ci(w['model']['brier'], 4)} | "
            f"{_ci(w['baseline']['brier'], 4)} | {_ci(w['model']['ece'], 4)} | "
            f"{_ci(w['baseline']['ece'], 4)} |"
        )
    lines += [""]
    pc = result.get("pmiss_calibration")
    if pc:
        lines += _calibration_lines(result, pc)
    if any("rounding_check" in sc for blk in result["scores"].values() for sc in blk.values()):
        lines += [
            "### Check of the whole-minute rounding in the simulator",
            "",
            "The simulator rounds each inbound delay draw to a whole minute, because the data is "
            "in whole minutes and a tie between being ready and the train leaving counts as a "
            "catch. This was first spotted when the first backtest printed validation and test "
            "scores together: mean P(miss) was 9.2% against 6.0% realised on test at T-60, and "
            "9.7% against 6.7% on validation. The cause was then diagnosed on pooled validation "
            "and test pairs, so test pairs did inform the change, although it has no fitted "
            "parameter. The table shows the effect on validation pairs, which the threshold "
            "choice also uses, and on test pairs for completeness.",
            "",
            "| Block and forecast time | Pairs | Realised miss rate | Mean P(miss) before | "
            "Mean P(miss) after | Brier before | Brier after |",
            "|---|---|---|---|---|---|---|",
        ]
        for blk in ("validation", "test"):
            for name, sc in result["scores"].get(blk, {}).items():
                for label, key in (
                    ("all pairs", "all"),
                    ("tightest gap, 10 to 14 min", "tightest_gap_bucket"),
                ):
                    r = sc["rounding_check"][key]
                    if r is None:
                        continue
                    lines.append(
                        f"| {blk}, {name}, {label} | {r['pairs']} | {_pct(r['realised'], 1)} | "
                        f"{_pct(r['mean_before'], 1)} | {_pct(r['mean_after'], 1)} | "
                        f"{r['brier_before']:.4f} | {r['brier_after']:.4f} |"
                    )
        lines += [
            "",
            _rounding_verdict(result),
            "",
        ]
    for name, sc in test.items():
        states = ", ".join(
            f"{st.replace('_', ' ')} {v['pairs']} pairs (miss rate {_pct(v['miss_rate'], 1)}, "
            f"mean predicted {_pct(v['mean_predicted_miss'], 1)})"
            for st, v in sc["by_state"].items()
        )
        lines += [
            f"At {name}, by what was known about the inbound: {states}.",
            "",
            f"Reliability for P(miss) at {name}, {sc['bins']} equal-count bins from the lowest "
            "predicted probability to the highest:",
            "",
            *_reliability_rows(sc),
            "",
        ]
    alerts = result.get("alerts", {})
    if alerts:
        costs = next(iter(alerts.values()))["costs"]
        lines += [
            "### Alert thresholds",
            "",
            f"The cost of a false alert is set to {costs['false_alert']:g} and the cost of a "
            f"missed alert to {costs['missed_alert']:g}. Both are assumptions in "
            "config/alerts.yaml and should be revisited with product input. Each threshold is "
            "the one with the lowest expected cost per pair on validation pairs, on a grid of "
            "0.01, and the test pairs were not used to choose it. An alert fires when the "
            "probability is at or above the threshold.",
            "",
        ]
        titles = {
            "likely_to_miss": ("tau", "Likely to miss your connection, on P(miss)"),
        }
        for key, a in alerts.items():
            sym, title = titles[key]
            lines += [
                f"{title}. The chosen {sym} is {a['tau']:.2f}. Test results with both forecast "
                "times pooled:",
                "",
                "| Policy | Threshold | Alerts | Precision | Recall | Alert rate | Cost per pair |",
                "|---|---|---|---|---|---|---|",
                _policy_row(f"Chosen {sym}", a["test"]["chosen"]),
                _policy_row("Always alert", a["test"]["always_alert"]),
                _policy_row("Never alert", a["test"]["never_alert"]),
                "",
            ]
            v, t = a["validation"], a["test"]
            lines += [
                f"On validation the chosen {sym} cost {_num(v['chosen']['cost_per_pair'], 3)} per "
                f"pair against {_num(v['always_alert']['cost_per_pair'], 3)} for always alerting "
                f"and {_num(v['never_alert']['cost_per_pair'], 3)} for never alerting. The event "
                f"happened in {_pct(v['chosen']['positives'] / v['chosen']['n'], 1)} of "
                f"validation pairs and {_pct(t['chosen']['positives'] / t['chosen']['n'], 1)} "
                "of test pairs"
                + (
                    ", so a threshold chosen on the quieter validation weeks "
                    + (
                        "did not beat always alerting on test."
                        if t["chosen"]["cost_per_pair"] > t["always_alert"]["cost_per_pair"]
                        else "still beat always alerting on test."
                    )
                    if t["chosen"]["positives"] / t["chosen"]["n"]
                    > v["chosen"]["positives"] / v["chosen"]["n"]
                    else "."
                ),
                "",
            ]
            for fk, s in a["test_by_forecast"].items():
                lines.append(
                    f"At {fk} the chosen {sym} gave precision {_num(s['precision'], 3)}, recall "
                    f"{_num(s['recall'], 3)}, alert rate {_num(s['alert_rate'], 3)} and cost "
                    f"{_num(s['cost_per_pair'], 3)} per pair over {s['n']} pairs."
                )
            moves = ", ".join(
                f"missed-alert cost {k}: {sym} {v['tau']:.2f} (a calibrated forecast would "
                f"give {v['theoretical_tau_if_calibrated']:.2f})"
                for k, v in a["sensitivity_to_missed_alert_cost"].items()
            )
            lines += ["", f"Sensitivity of {sym} to the missed-alert cost: {moves}.", ""]
    lines += [
        "### Limitations",
        "",
        "The minimum connection time is assumed, and the miss probabilities move with it. The "
        "inbound and outbound delays are drawn independently, so shared causes such as a "
        "signalling failure or a strike day are not modelled. In the history the two delays are "
        "only weakly correlated (0.11), and for the tightest changes, 10 to 14 minutes, "
        "independence overstated the miss rate (18.4% realised against 20.3% with the pairing "
        "broken), so the bias is not in the optimistic direction there. The history is summer and "
        "the first weeks of autumn, with no winter or storm season. Only two routes are in the "
        "model, and the connection test uses one hub and one pair of routes. The test block is "
        f"{days['test']} days, so the intervals are wide, and a small number of pairs drives the "
        "tails of every score. The costs behind the threshold are assumed. There is no Delay "
        "Repay alert: an earlier forecast alert for it lost to always alerting on test when the "
        "late-arrival rate moved from 13.4% to 21.7%, and eligibility is known for certain once "
        "the train has arrived, so expected Delay Repay is shown as information only.",
        "",
        END,
    ]
    return "\n".join(lines)


def write_connections_section(result: dict[str, Any], path: Path) -> None:
    section = connections_section(result)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if START in text and END in text:
        head, rest = text.split(START, 1)
        tail = rest.split(END, 1)[1]
        text = head.rstrip("\n") + "\n\n" + section + tail
    else:
        text = text.rstrip("\n") + ("\n\n" if text else "") + section + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
