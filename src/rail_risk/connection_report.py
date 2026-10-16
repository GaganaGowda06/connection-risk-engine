"""The connections section of reports/model_results.md, written from the backtest result.

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

    `outcome` is "missed" or "delay15". delay15 is 1 when the realised final arrival was at
    least 15 minutes late or the passenger was stranded, 0 when it was less, and unknown when no
    final arrival exists. Pairs with an unknown outcome are left out.
    """
    df = pred[pred["missed"].notna()].copy()
    df["delay15"] = np.where(
        df["stranded"], 1.0, np.where(df["final_delay"].notna(), df["final_delay"] >= 15, np.nan)
    )
    df = df[df[outcome].notna()]

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
        "inside the predicted 80% range, counting only pairs that reached Euston.",
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
            "may_be_owed_delay_repay": (
                "tau2",
                "You may be owed Delay Repay, on P(final arrival at least 15 minutes late)",
            ),
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
        "signalling failure or a strike day are not modelled, and the model's miss probabilities "
        "are optimistic when delays are correlated. The history is summer and the first weeks of "
        "autumn, with no winter or storm season. Only two routes are in the model, and the "
        "connection test uses one hub and one pair of routes. The test block is "
        f"{days['test']} days, so the intervals are wide, and a small number of pairs drives the "
        "tails of every score. The Delay Repay alert counts a stranded passenger as at least 15 "
        "minutes late, which is an assumption. The costs behind both thresholds are assumed.",
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
