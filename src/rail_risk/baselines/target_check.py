"""Target sanity check, run on every build: the target distribution per split block and route.

Stage 2 once fitted on targets that were off by a day for about 2.5 percent of rows, and nothing
looked at the target distribution. This check fails loudly (TargetCheckError) on:
  duplicate targets        the same (rid, route, horizon) twice
  day-sized offsets        a target within an hour of a multiple of 24 hours (and over 12 hours)
  impossible lateness      under -180 minutes or over 1,000 minutes
  systematic offsets       over 10 percent of a route's rows under -30 minutes in one block, or over
                           5 percent over 120 minutes (needs at least 200 rows)
It also records warnings that do not fail: a route with over 1 percent of rows under -30 minutes
(the Caledonian Sleeper services on glc-eus arrive about an hour before the public time).
Targets are in seconds (target_late_s), rows without a target are ignored.
"""

import numpy as np
import pandas as pd

KEY = ["rid", "route", "horizon_min"]
DAY_MIN = 1440.0
MAX_ABS_MIN = 720.0
IMPOSSIBLE = (-180.0, 1000.0)
SYSTEMATIC_EARLY = (-30.0, 0.10)
SYSTEMATIC_LATE = (120.0, 0.05)
WARN_EARLY_SHARE = 0.01
MIN_ROWS = 200


class TargetCheckError(RuntimeError):
    pass


def check_targets(df: pd.DataFrame, blocks: dict[str, pd.DataFrame] | None = None) -> dict:
    """Report on df (columns rid, route, horizon_min, target_late_s) or on named blocks.

    Returns {'failures': [...], 'warnings': [...], 'rows': {block: n}}; never raises."""
    blocks = blocks if blocks is not None else {"all": df}
    failures: list[str] = []
    warnings: list[str] = []
    rows: dict[str, int] = {}
    for name, part in blocks.items():
        part = part[part["target_late_s"].notna()]
        rows[name] = int(len(part))
        if part.empty:
            continue
        y = part["target_late_s"].to_numpy(dtype="float64") / 60.0
        dups = int(part.duplicated(KEY).sum())
        if dups:
            failures.append(f"{name}: {dups} duplicate targets")
        big = np.abs(y) >= MAX_ABS_MIN
        near = np.abs(np.abs(y) - np.round(np.abs(y) / DAY_MIN) * DAY_MIN) < 60.0
        day_sized = big & near & (np.abs(y) > MAX_ABS_MIN)
        if day_sized.any():
            failures.append(f"{name}: {int(day_sized.sum())} day-sized offsets")
        if big.any():
            failures.append(f"{name}: {int(big.sum())} targets of 12 hours or more")
        bad = (y < IMPOSSIBLE[0]) | (y > IMPOSSIBLE[1])
        if bad.any():
            failures.append(f"{name}: {int(bad.sum())} impossible lateness values")
        by = part.assign(y=y).groupby("route")["y"]
        for route, g in by:
            if len(g) < MIN_ROWS:
                continue
            early = float((g < SYSTEMATIC_EARLY[0]).mean())
            late = float((g > SYSTEMATIC_LATE[0]).mean())
            if early > SYSTEMATIC_EARLY[1]:
                failures.append(f"{name}/{route}: {early:.1%} of rows under -30 minutes")
            elif early > WARN_EARLY_SHARE:
                warnings.append(f"{name}/{route}: {early:.1%} of rows under -30 minutes")
            if late > SYSTEMATIC_LATE[1]:
                failures.append(f"{name}/{route}: {late:.1%} of rows over 120 minutes")
    return {"failures": failures, "warnings": warnings, "rows": rows}


def assert_targets_ok(df: pd.DataFrame, blocks: dict[str, pd.DataFrame] | None = None) -> dict:
    report = check_targets(df, blocks)
    if report["failures"]:
        raise TargetCheckError("target sanity check failed: " + "; ".join(report["failures"][:6]))
    return report
