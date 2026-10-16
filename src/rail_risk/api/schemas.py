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
    stale_parse: bool = Field(
        default=False,
        description="Closed hours of Darwin capture are waiting to be parsed, so the parsed "
        "tables, schedules included, are behind the raw files",
    )
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
    stale_parse: bool = False
    parse_lag_hours: float | None = None


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


HHMM_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"


class ConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inbound_rid: str = Field(pattern=RID_PATTERN, description="The Aston to New Street service")
    outbound_departure: str | None = Field(
        default=None,
        pattern=HHMM_PATTERN,
        description="Scheduled New Street departure, HH:MM. Omitted: the first feasible train",
    )
    fare: float | None = Field(default=None, ge=0, le=1000)
    ticket: Literal["single", "return"] = "single"


class InboundOut(BaseModel):
    rid: str
    scheduled_arrival: str | None = Field(description="Scheduled arrival at the hub, HH:MM")
    state: Literal["arrived", "approaching", "due_or_late", "cancelled"]
    minutes_to_arrival: int | None
    expected_delay_minutes: float | None = Field(
        description="Known delay when it has arrived, otherwise the model's median"
    )


class OutboundOut(BaseModel):
    rid: str
    toc: str | None
    scheduled_departure: str
    scheduled_arrival: str = Field(description="Scheduled arrival at the final stop, HH:MM")
    scheduled_gap_minutes: float
    chosen_automatically: bool
    alternatives: list[str] = Field(description="Later departures that would carry a missed pass")


class ConnectionTimeOut(BaseModel):
    station: str
    minutes: int
    status: str
    comment: str | None = None


class SensitivityRow(BaseModel):
    minutes: int
    feasible: bool
    p_miss: float | None
    arrival_range_80: str | None
    expected_payout_gbp: float | None


class ConnectionDelayRepay(DelayRepayOut):
    stranded_assumption: str


class ConnectionResponse(BaseModel):
    request_id: str
    status: Literal["ok", "refused", "cancelled"]
    inbound: InboundOut | None
    outbound: OutboundOut | None
    refusal: str | None = None
    first_feasible_departure: str | None = None
    p_miss: float | None
    alert: bool | None = Field(description="True when P(miss) is at least the alert threshold")
    alert_threshold: float | None
    p_stranded: float | None
    arrival_range_80: str | None = Field(description='Final arrival, for example "11:42 to 11:55"')
    median_arrival: str | None
    delay_repay: ConnectionDelayRepay | None
    connection_time: ConnectionTimeOut | None
    sensitivity: list[SensitivityRow]
    assumptions: list[str]
    model_run_id: str | None
    model_version: str | None
    data_quality: DataQuality
