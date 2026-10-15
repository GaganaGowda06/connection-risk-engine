"""Europe/London wall-clock handling shared by the HSP and Darwin parsers.

Daylight saving: an ambiguous local time (the repeated hour when clocks go back, for example
2026-10-25 01:30) is read as the first occurrence (BST) unless that would put the time before the
previous event of the same service, in which case it is read as the second (GMT). A
non-existent local time (clocks go forward, for example 2026-03-29 01:30) is read with the
offset in force before the change, so it lands at the same UTC instant as the wall clock
(01:30Z, which is 02:30 BST), a shift forward by the length of the gap.
"""

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
# a backwards step in a chain of planned times larger than this is midnight, smaller is a
# clock change or noise
ROLLOVER_MIN_DROP = timedelta(hours=3)


def to_utc(local: datetime, floor: datetime | None = None) -> datetime:
    """Europe/London wall clock (naive) to aware UTC, never earlier than `floor` if possible."""
    first = local.replace(tzinfo=LONDON, fold=0).astimezone(UTC)
    second = local.replace(tzinfo=LONDON, fold=1).astimezone(UTC)
    # ambiguous: the second occurrence is later in UTC. Non-existent: fold 0 already uses the
    # pre-change offset, which is the documented reading.
    if second > first and floor is not None and first < floor:
        return second
    return first


def to_utc_near(local: datetime, reference: datetime) -> datetime:
    """Like to_utc, but in an ambiguous hour pick the occurrence closest to `reference`."""
    first = local.replace(tzinfo=LONDON, fold=0).astimezone(UTC)
    second = local.replace(tzinfo=LONDON, fold=1).astimezone(UTC)
    if second > first and abs(second - reference) < abs(first - reference):
        return second
    return first


def nearest_day(t: time, anchor: datetime) -> datetime:
    """The datetime with clock time t on the day (anchor-1, anchor, anchor+1) closest to anchor."""
    base = datetime.combine(anchor.date(), t)
    options = [base + timedelta(days=d) for d in (-1, 0, 1)]
    return min(options, key=lambda dt: abs(dt - anchor))


def parse_clock(text: str | None) -> time | None:
    """HH:MM or HH:MM:SS, or None for a missing value."""
    if not text:
        return None
    parts = text.split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() and len(p) == 2 for p in parts):
        raise ValueError(f"bad clock time {text!r}")
    h, m = int(parts[0]), int(parts[1])
    s = int(parts[2]) if len(parts) == 3 else 0
    if h > 23 or m > 59 or s > 59:
        raise ValueError(f"bad clock time {text!r}")
    return time(h, m, s)
