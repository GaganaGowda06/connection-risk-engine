"""Data-quality badges: what the API says about its own inputs, in plain words."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Badge:
    level: Literal["warning", "info"]
    title: str
    text: str


def badges(risk: dict[str, Any]) -> list[Badge]:
    """One badge for each data-quality condition that applies to this answer."""
    dq = risk.get("data_quality", {})
    out: list[Badge] = []
    if dq.get("stale_live_data"):
        age = dq.get("live_data_age_minutes")
        how_old = f"{age:.0f} minutes old" if age is not None else "out of date"
        out.append(
            Badge(
                "warning",
                "Live data is old",
                f"The live train feed is {how_old}, so this forecast ignores where the train is "
                "right now.",
            )
        )
    if dq.get("model_fallback"):
        out.append(
            Badge(
                "warning",
                "Backup forecast",
                "The main model is not available, so this uses a simple average of past trains.",
            )
        )
    snapped = dq.get("horizon_snapped_from_minutes")
    if snapped is not None and dq.get("horizon_used_minutes") is not None:
        out.append(
            Badge(
                "info",
                "Forecast point rounded",
                f"The model looks at fixed points, so for a train {snapped:.0f} minutes from "
                f"arrival it used its {dq['horizon_used_minutes']} minute view.",
            )
        )
    if dq.get("route_in_training") is False:
        out.append(
            Badge(
                "warning",
                "Route not checked",
                "The model was not trained on this route, so its accuracy here is unknown.",
            )
        )
    repay = risk.get("delay_repay") or {}
    if repay.get("toc_status") in ("unverified", "unknown"):
        who = repay.get("toc") or "this operator"
        out.append(
            Badge(
                "info",
                "Delay Repay rules unverified",
                f"We could not verify the Delay Repay rules for {who}, so no amount is shown.",
            )
        )
    return out
