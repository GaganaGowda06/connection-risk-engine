"""Parse HSP serviceDetails into one row per service per stop, and validate at the boundary.

Times in HSP are HHMM strings in UK local time, with an empty string for missing values. They
are stored here as naive local timestamps.
"""

from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
import pandera.pandas as pa

COLUMNS = [
    "rid",
    "date_of_service",
    "toc_code",
    "seq",
    "location",
    "sched_arr",
    "sched_dep",
    "actual_arr",
    "actual_dep",
    "late_canc_reason",
    "candidate_cancelled",
]
MAX_ACTUAL_DEVIATION = pd.Timedelta(hours=6)


def _hhmm(value: str | None) -> time | None:
    if not value:
        return None
    if len(value) != 4 or not value.isdigit():
        raise ValueError(f"bad time {value!r}")
    return time(int(value[:2]), int(value[2:]))


def _nearest(t: time, anchor: datetime) -> datetime:
    """The datetime with clock time t on the day (anchor-1, anchor, anchor+1) closest to anchor."""
    base = datetime.combine(anchor.date(), t)
    options = [base + timedelta(days=d) for d in (-1, 0, 1)]
    return min(options, key=lambda dt: abs(dt - anchor))


def parse_details(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for one serviceDetails response. Returns [] when the response has no locations."""
    attrs = payload.get("serviceAttributesDetails") or {}
    locations = attrs.get("locations") or []
    if not locations:
        return []
    service_date = date.fromisoformat(attrs["date_of_service"])
    last = locations[-1]
    candidate_cancelled = not (last.get("actual_ta") or last.get("actual_td"))

    rows: list[dict[str, Any]] = []
    sched_prev = datetime.combine(service_date, time(0, 0))
    actual_prev: datetime | None = None
    for seq, loc in enumerate(locations):
        resolved: dict[str, datetime | None] = {}
        # scheduled times run forward along the stop list, so a drop means midnight passed
        for key, field in (("sched_arr", "gbtt_pta"), ("sched_dep", "gbtt_ptd")):
            t = _hhmm(loc.get(field))
            if t is None:
                resolved[key] = None
                continue
            dt = datetime.combine(sched_prev.date(), t)
            if dt < sched_prev:
                dt += timedelta(days=1)
            resolved[key] = dt
            sched_prev = dt
        # actual times are placed on the day nearest their scheduled counterpart, then nearest
        # the previous actual time when there is no scheduled time
        for key, field, sched_key in (
            ("actual_arr", "actual_ta", "sched_arr"),
            ("actual_dep", "actual_td", "sched_dep"),
        ):
            t = _hhmm(loc.get(field))
            if t is None:
                resolved[key] = None
                continue
            anchor = resolved[sched_key] or actual_prev or sched_prev
            dt = _nearest(t, anchor)
            resolved[key] = dt
            actual_prev = dt
        rows.append(
            {
                "rid": attrs["rid"],
                "date_of_service": pd.Timestamp(service_date),
                "toc_code": attrs.get("toc_code"),
                "seq": seq,
                "location": loc["location"],
                **resolved,
                "late_canc_reason": loc.get("late_canc_reason") or None,
                "candidate_cancelled": candidate_cancelled,
            }
        )
    return rows


def to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ("date_of_service", "sched_arr", "sched_dep", "actual_arr", "actual_dep"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    df["seq"] = df["seq"].astype("int64")
    df["candidate_cancelled"] = df["candidate_cancelled"].astype(bool)
    return df


def _within_6h(actual: str, sched: str) -> pa.Check:
    def check(df: pd.DataFrame) -> pd.Series:
        diff = (df[actual] - df[sched]).abs()
        return diff.isna() | (diff <= MAX_ACTUAL_DEVIATION)

    return pa.Check(check, name=f"{actual}_within_6h_of_{sched}")


STOPS_SCHEMA = pa.DataFrameSchema(
    {
        "rid": pa.Column(str, pa.Check.str_matches(r"^\d{15}$"), nullable=False),
        "date_of_service": pa.Column(pa.DateTime, nullable=False),
        "toc_code": pa.Column(str, pa.Check.str_matches(r"^[A-Z0-9]{2}$"), nullable=False),
        "seq": pa.Column(int, pa.Check.ge(0), nullable=False),
        "location": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{3}$"), nullable=False),
        "sched_arr": pa.Column(pa.DateTime, nullable=True),
        "sched_dep": pa.Column(pa.DateTime, nullable=True),
        "actual_arr": pa.Column(pa.DateTime, nullable=True),
        "actual_dep": pa.Column(pa.DateTime, nullable=True),
        "late_canc_reason": pa.Column(str, pa.Check.str_matches(r"^\d+$"), nullable=True),
        "candidate_cancelled": pa.Column(bool, nullable=False),
    },
    checks=[_within_6h("actual_arr", "sched_arr"), _within_6h("actual_dep", "sched_dep")],
    unique=["rid", "location", "seq"],
    strict=True,
    coerce=False,
)


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into valid and quarantined. A failing row quarantines its whole service.

    A service missing one stop would mislead the cancellation label and the delay features, so
    all stops of a rid with any failure go to quarantine, with the reasons joined in one column.
    """
    if df.empty:
        return df, df.assign(quarantine_reason=pd.Series(dtype=str))
    df = df.reset_index(drop=True)
    try:
        STOPS_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases
        reasons: dict[str, set[str]] = {}
        for _, f in failures.iterrows():
            if f["schema_context"] == "DataFrameSchema":
                label = str(f["check"])
            else:
                label = f"{f['column']}: {f['check']}"
            if pd.isna(f["index"]):
                targets = list(df["rid"].unique())
            else:
                targets = [df.loc[int(f["index"]), "rid"]]
            for rid in targets:
                reasons.setdefault(rid, set()).add(label)
        bad = df["rid"].isin(reasons)
        quarantined = df[bad].copy()
        quarantined["quarantine_reason"] = quarantined["rid"].map(
            lambda r: "; ".join(sorted(reasons[r]))
        )
        return df[~bad].reset_index(drop=True), quarantined.reset_index(drop=True)
    return df, df.iloc[0:0].assign(quarantine_reason=pd.Series(dtype=str))
