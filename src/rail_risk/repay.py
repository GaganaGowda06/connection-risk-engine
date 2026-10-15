"""Delay Repay rules and expected payout.

Rules come from `config/delay_repay.yaml`, keyed by TOC code and read only from each operator's
own page. A TOC whose page could not be read is marked unverified and has no tiers, and asking it
for a payout is an error, never a number. The fare is always supplied by the caller because there
is no public fare API.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

TicketType = Literal["single", "return"]
TICKETS = ("single", "return")


class RepayError(ValueError):
    """The payout cannot be priced: unknown or unverified TOC, bad fare or bad probabilities."""


class Tier(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    from_min: int
    to_min: int | None  # exclusive; None for the open-ended last tier
    single: float
    return_: float

    @model_validator(mode="before")
    @classmethod
    def _rename(cls, data: dict) -> dict:
        data = dict(data)
        if "return" in data:
            data["return_"] = data.pop("return")
        return data

    @field_validator("single", "return_")
    @classmethod
    def _share(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"payout share must be between 0 and 1, got {v}")
        return v

    @model_validator(mode="after")
    def _ordered(self) -> "Tier":
        if self.from_min < 0 or (self.to_min is not None and self.to_min <= self.from_min):
            raise ValueError(f"tier {self.from_min} to {self.to_min} is not a positive range")
        return self

    @property
    def label(self) -> str:
        return f"{self.from_min}+" if self.to_min is None else f"{self.from_min}-{self.to_min - 1}"

    def share(self, ticket: str) -> float:
        return self.single if ticket == "single" else self.return_


class Operator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    status: Literal["verified", "unverified"]
    source_url: str
    checked: date
    page_last_updated: date | None = None
    claim_window_days: int | None = None
    measured_at: str | None = None
    note: str | None = None
    tiers: list[Tier] = []

    @model_validator(mode="after")
    def _tiers_match_status(self) -> "Operator":
        if self.status == "unverified":
            if self.tiers:
                raise ValueError("an unverified operator must not carry tiers")
            return self
        if not self.tiers:
            raise ValueError("a verified operator needs tiers")
        for a, b in zip(self.tiers, self.tiers[1:], strict=False):
            if a.to_min is None:
                raise ValueError("only the last tier may be open ended")
            if b.from_min < a.to_min:
                raise ValueError(f"tiers overlap at {b.from_min} minutes")
            if b.from_min > a.to_min:
                raise ValueError(f"tiers are not contiguous between {a.to_min} and {b.from_min}")
        if self.tiers[-1].to_min is not None:
            raise ValueError("the last tier must be open ended")
        return self


class DelayRepayRules(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    operators: dict[str, Operator]

    def operator(self, toc: str) -> Operator:
        """The verified operator for a TOC code, or a RepayError saying why not."""
        op = self.operators.get(toc)
        if op is None:
            raise RepayError(f"no Delay Repay rules for TOC {toc!r}")
        if op.status != "verified":
            raise RepayError(
                f"Delay Repay rules for TOC {toc!r} ({op.name}) are unverified, "
                f"so no payout is priced; see {op.source_url}"
            )
        return op


def load_rules(path: Path | str = "config/delay_repay.yaml") -> DelayRepayRules:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Delay Repay rules file not found: {path}")
    return DelayRepayRules.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class TierPayout:
    tier: str
    probability: float
    share: float
    payout_if_in_tier: float
    expected: float


@dataclass(frozen=True)
class ExpectedRepay:
    toc: str
    ticket: str
    fare: float
    expected_payout: float
    tiers: tuple[TierPayout, ...]
    p_no_payout: float


def expected_repay(
    probs: np.ndarray | list[float],
    lower_edges: tuple[int, ...],
    toc: str,
    ticket: str,
    fare: float,
    rules: DelayRepayRules,
) -> ExpectedRepay:
    """Expected Delay Repay in pounds for one journey.

    `probs` are band probabilities for the bands whose lower edges are `lower_edges`. Every
    tier edge must be a band edge, so each band lies wholly inside one tier or below the first.
    A zero fare prices to 0.0; a negative fare, an unverified or unknown TOC, an unknown ticket
    type or probabilities that are not a distribution raise RepayError.
    """
    if ticket not in TICKETS:
        raise RepayError(f"ticket type must be one of {TICKETS}, got {ticket!r}")
    if not np.isfinite(fare) or fare < 0:
        raise RepayError(f"fare must be a non-negative number of pounds, got {fare!r}")
    op = rules.operator(toc)
    p = np.asarray(probs, dtype=float)
    if p.shape != (len(lower_edges),):
        raise RepayError(f"expected {len(lower_edges)} band probabilities, got shape {p.shape}")
    if (p < -1e-9).any() or abs(p.sum() - 1.0) > 1e-6:
        raise RepayError("band probabilities must be non-negative and sum to 1")
    edges = [*lower_edges, None]
    for tier in op.tiers:
        for edge in (tier.from_min, tier.to_min):
            if edge is not None and edge not in lower_edges:
                raise RepayError(f"tier edge {edge} is not a delay band edge {lower_edges}")

    tiers = []
    covered = 0.0
    for tier in op.tiers:
        inside = [
            k
            for k in range(len(lower_edges))
            if lower_edges[k] >= tier.from_min
            and (tier.to_min is None or (edges[k + 1] is not None and edges[k + 1] <= tier.to_min))
        ]
        prob = float(p[inside].sum()) if inside else 0.0
        covered += prob
        pay = tier.share(ticket) * fare
        tiers.append(TierPayout(tier.label, prob, tier.share(ticket), pay, prob * pay))
    total = float(sum(t.expected for t in tiers))
    return ExpectedRepay(
        toc, ticket, float(fare), total, tuple(tiers), p_no_payout=float(1.0 - covered)
    )
