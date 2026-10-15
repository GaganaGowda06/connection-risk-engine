"""Request and response models for the risk API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RID_PATTERN = r"^\d{15}$"


class RiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rid: str = Field(pattern=RID_PATTERN, description="Darwin and HSP service id, 15 digits")
    fare: float | None = Field(
        default=None, ge=0, le=1000, description="Ticket price in pounds; always supplied"
    )
    ticket: Literal["single", "return"] = "single"
    route: str | None = Field(default=None, max_length=40)


class DataQuality(BaseModel):
    live_data_age_minutes: float | None
    stale_live_data: bool
    model_fallback: bool
    horizon_snapped_from_minutes: float | None = Field(
        description="Real minutes to arrival when the model was given a different trained horizon"
    )
    horizon_used_minutes: int | None = Field(
        default=None, description="The horizon the model was given, when a model was used"
    )
    route_in_training: bool
    schedule_known: bool
    features_age_seconds: float | None
    notes: list[str]


class DelayRepayOut(BaseModel):
    toc: str | None
    toc_status: Literal["verified", "unverified", "unknown"]
    amount_gbp: float | None
    reason: str | None = Field(description="Why no amount is given, when it is null")
    caveat: str | None = None


class RiskResponse(BaseModel):
    request_id: str
    rid: str
    route_id: str
    status: Literal["running", "not_started", "arrived", "cancelled"]
    toc: str | None
    scheduled_arrival: str | None = Field(description="Local clock time HH:MM")
    arrival_range_80: str | None = Field(description='For example "11:42 to 11:55"')
    median_arrival: str | None
    median_delay_minutes: float | None
    p_delay_at_least_15: float | None
    p_delay_at_least_30: float | None
    p_delay_at_least_60: float | None
    minutes_to_arrival_used: float | None
    cancellation_reason: str | None = None
    delay_repay: DelayRepayOut | None
    model_run_id: str | None
    model_version: str | None
    data_quality: DataQuality


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_alias: str
    model_run_id: str | None
    model_fallback: bool
    darwin_data_age_minutes: float | None
    stale_live_data: bool
    watched_services: int


class Warming(BaseModel):
    status: Literal["warming"] = "warming"
    request_id: str
    rid: str
    retry_after_seconds: int
    detail: str


class ServiceItem(BaseModel):
    rid: str
    route_id: str
    toc: str | None
    origin: str
    scheduled_departure: str
    scheduled_arrival: str
    minutes_to_arrival: int
    started: bool
    cancelled: bool


class RouteItem(BaseModel):
    id: str
    origin: str
    destination: str
    model_trained_on_route: bool
