# ruff: noqa: S608  (the f-strings only interpolate module constants, never data)
"""Point-in-time features for the route-agnostic model, one service date at a time (DuckDB).

Every feature for a snapshot at time T uses only messages with message_time <= T (plan data uses
the first schedule version, which the snapshot table already requires to be known at T). Nothing
here names a route, a station, a TIPLOC or an operator as a value that reaches the model: stations
appear only as join keys, and the features are numbers. The model sees FEATURES and nothing else.

Families: train state, timetable slack, what is ahead (same stop, destination, next stop, network,
cancellations), associations, calendar. The slack feature needs the fastest observed run time over
the same stops from train dates, which is a second pass (fastest_table, then add_slack).
"""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.baselines.state import UPDATE_COLS
from rail_risk.ingest.pport_extract import train_pool_table

WINDOW_MIN = 30
LATE5_S = 300
MIN_PAIR_OBS = 20

STATE = [
    "late_now_min", "late_chg_1_min", "late_chg_3_min", "late_since_start_min", "n_actuals",
    "fc_late_min", "fc_minus_now_min", "minutes_since_last_actual", "horizon_min",
]  # fmt: skip
REMAINING = ["stops_remaining", "sched_min_remaining"]
SLACK = ["fastest_min_remaining", "slack_min", "late_minus_slack_min"]
AHEAD = [
    "ahead_n", "ahead_late_mean_min", "ahead_late5_share",
    "dest_n", "dest_late_mean_min", "dest_late5_share", "dest_cancel_n",
    "next_n", "next_late_mean_min", "next_late5_share", "next_cancel_n",
    "net_n", "net_late5_share", "net_late_mean_min", "net_cancel_n",
]  # fmt: skip
ASSOC = [
    "assoc_in_n", "assoc_in_main_late_min", "assoc_in_main_arrived", "assoc_in_turn_min",
    "assoc_out_n", "assoc_join_n", "assoc_divide_n", "assoc_next_n",
]  # fmt: skip
CALENDAR = ["arr_hour_local", "weekday", "is_bank_holiday"]
# lateness of the same route at the same arrival hour over the previous SLOT_DAYS days (strictly
# earlier service dates): a route's own recent history, so unavailable on a route's first days
HISTORY = [
    "slot_median_min", "slot_p90_min", "slot_late15_share", "slot_n",
    "recent_recovery_min", "recent_final_min", "recent_n",
]  # fmt: skip
SLOT_DAYS = 28
SLOT_MIN_N = 5
# a route's running drifts over weeks (bhm-liv gained about three minutes between January and
# October), so a short window of the route's own recent recovery is also given to the model
RECENT_DAYS = 7
RECENT_MIN_N = 20
FEATURES = [*STATE, *REMAINING, *SLACK, *AHEAD, *ASSOC, *CALENDAR, *HISTORY]
# column names the model must never see, whatever their form (tests/test_network_model.py)
BANNED_SUBSTRINGS = ("route", "rid", "tpl", "tiploc", "crs", "toc", "operator", "hub")
KEY = ["rid", "route", "horizon_min"]
INTERNAL = ["from_tpl", "from_event", "dest_tpl"]
SNAP_IN = [
    "rid", "route", "horizon_min", "snap_time", "dest_tpl", "pt_arr", "last_actual_late_s",
    "fc_late_s",
]  # fmt: skip


def plan_sql(event: str, p: str = "") -> str:
    """Planned time of an event kind: the public time, else working time (same rule as state.py)."""
    return (
        f"CASE {event} WHEN 'arr' THEN coalesce({p}pt_arr, {p}wt_arr) "
        f"WHEN 'dep' THEN coalesce({p}pt_dep, {p}wt_dep) ELSE {p}wt_pass END"
    )


def bank_holidays(path: Path = Path("config/bank_holidays.json")) -> set[date]:
    return {date.fromisoformat(e["date"]) for e in json.loads(path.read_text())["events"]}


def _events(root: Path, ssd: date) -> "object":
    """Actual events reported on ssd-1 and ssd, first report per (rid, tpl, event)."""
    import pyarrow.dataset as ds

    cols = [c for c in UPDATE_COLS if c != "est"]
    days = [ssd - timedelta(days=1), ssd]
    return train_pool_table(root, "stop_updates", cols, days, where=ds.field("actual").is_valid())


def _without(t, gap: tuple[pd.Timestamp, pd.Timestamp] | None):
    import pyarrow.compute as pc

    if gap is None or t.num_rows == 0:
        return t
    lo, hi = (pd.Timestamp(x).to_pydatetime() for x in gap)
    inside = pc.and_(pc.greater_equal(t["message_time"], lo), pc.less(t["message_time"], hi))
    return t.filter(pc.invert(inside))


def _as_of(t, cutoff: pd.Timestamp | None):
    """Drop messages after the cutoff (the leakage check recomputes one row on a truncated feed)."""
    import pyarrow.compute as pc

    if cutoff is None or t.num_rows == 0:
        return t
    return t.filter(pc.less_equal(t["message_time"], pd.Timestamp(cutoff).to_pydatetime()))


def build_day(
    root: Path,
    ssd: date,
    snaps: pd.DataFrame,
    holidays: set[date],
    as_of: pd.Timestamp | None = None,
    gap: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Feature rows for the snapshots of one day (schedule-known rows only). `as_of` is only for
    the leakage check: every table is cut at that time before the features are built. `gap` drops
    every message received inside a (start, end) window, to simulate a missing hour of feed."""
    import duckdb

    snaps = snaps[snaps["schedule_known"]].copy()
    if snaps.empty:
        return pd.DataFrame()
    updates = _without(_as_of(_events(root, ssd), as_of), gap)
    if updates.num_rows == 0:
        return pd.DataFrame()
    points = train_pool_table(
        root,
        "schedule_points",
        ["rid", "seq", "tpl", "pt_arr", "pt_dep", "wt_arr", "wt_dep", "wt_pass", "message_time"]
        + ["cancelled"],
        [ssd],
    )
    points = _without(_as_of(points, as_of), gap)
    assoc = train_pool_table(
        root,
        "associations",
        ["main_rid", "assoc_rid", "category", "tiploc", "is_cancelled", "is_deleted"]
        + ["main_wta", "assoc_wtd", "message_time"],
        [ssd],
    )
    assoc = _without(_as_of(assoc, as_of), gap)
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("u", updates)
    con.register("sp", points)
    con.register("asc_raw", assoc if assoc.num_rows else _empty_assoc())
    con.register("s", pd.DataFrame(snaps[SNAP_IN]).assign(ssd=ssd))
    con.execute(f"""
        CREATE TEMP TABLE ev AS
        SELECT rid, tpl, event, min(message_time) AS message_time,
               arg_min(actual, message_time) AS actual,
               arg_min(late_s, message_time) AS late_s, arg_min(pub, message_time) AS pub
        FROM (
          SELECT rid, tpl, event, message_time, actual,
                 date_diff('second', plan, actual) AS late_s,
                 (CASE event WHEN 'arr' THEN pt_arr WHEN 'dep' THEN pt_dep END) IS NOT NULL AS pub
          FROM (SELECT *, {plan_sql("event")} AS plan FROM u)
          WHERE plan IS NOT NULL
        ) GROUP BY rid, tpl, event
    """)
    con.execute("""
        CREATE TEMP TABLE own AS
        SELECT s.rid, s.route, s.horizon_min, count(*) AS n_act,
          list(e.late_s ORDER BY e.actual, e.message_time, e.tpl, e.event) AS ls,
          list(e.tpl ORDER BY e.actual, e.message_time, e.tpl, e.event) AS tp,
          list(e.event ORDER BY e.actual, e.message_time, e.tpl, e.event) AS evn,
          list(epoch(e.actual) ORDER BY e.actual, e.message_time, e.tpl, e.event) AS at
        FROM s JOIN ev e ON e.rid = s.rid AND e.message_time <= s.snap_time
        GROUP BY s.rid, s.route, s.horizon_min
    """)
    con.execute("""
        CREATE TEMP TABLE pts AS
        SELECT rid, seq, tpl, pt_arr, pt_dep, wt_arr, wt_dep, wt_pass,
               (pt_arr IS NOT NULL OR pt_dep IS NOT NULL) AS is_call,
               coalesce(pt_dep, pt_arr, wt_dep, wt_arr, wt_pass) AS plan_any
        FROM sp
        WHERE rid IN (SELECT rid FROM s)
        QUALIFY message_time = min(message_time) OVER (PARTITION BY rid)
    """)
    con.execute("""
        CREATE TEMP TABLE base AS
        SELECT s.*, o.n_act, o.ls, o.tp, o.evn, o.at,
          o.tp[-1] AS last_tpl, o.evn[-1] AS last_evn,
          (SELECT max(seq) FROM pts p WHERE p.rid = s.rid AND p.tpl = s.dest_tpl
             AND p.pt_arr IS NOT NULL) AS dseq
        FROM s LEFT JOIN own o USING (rid, route, horizon_min)
    """)
    con.execute("""
        CREATE TEMP TABLE pos AS
        SELECT b.rid, b.route, b.horizon_min, b.dseq,
          coalesce(
            (SELECT max(seq) FROM pts p WHERE p.rid = b.rid AND p.tpl = b.last_tpl
               AND p.seq <= b.dseq),
            (SELECT min(seq) - 1 FROM pts p WHERE p.rid = b.rid AND p.is_call)) AS fseq
        FROM base b
    """)
    con.execute("""
        CREATE TEMP TABLE rem AS
        SELECT x.rid, x.route, x.horizon_min, x.fseq,
          (SELECT count(*) FROM pts p WHERE p.rid = x.rid AND p.is_call
             AND p.seq > x.fseq AND p.seq <= x.dseq) AS stops_remaining,
          (SELECT tpl FROM pts p WHERE p.rid = x.rid AND p.is_call AND p.seq > x.fseq
             ORDER BY seq LIMIT 1) AS next_tpl,
          (SELECT plan_any FROM pts p WHERE p.rid = x.rid AND p.is_call AND p.seq > x.fseq
             ORDER BY seq LIMIT 1) AS first_plan,
          (SELECT tpl FROM pts p WHERE p.rid = x.rid AND p.is_call AND p.seq > x.fseq
             ORDER BY seq LIMIT 1) AS first_tpl
        FROM pos x
    """)
    # plan time at the stop of the last actual event, by event kind (same rule as lateness)
    con.execute(f"""
        CREATE TEMP TABLE planat AS
        SELECT b.rid, b.route, b.horizon_min,
          (SELECT {plan_sql("b.last_evn", "p.")}
             FROM pts p WHERE p.rid = b.rid AND p.tpl = b.last_tpl AND p.seq <= b.dseq
             ORDER BY p.seq DESC LIMIT 1) AS plan_last
        FROM base b WHERE b.n_act > 0
    """)
    con.execute("""
        CREATE TEMP TABLE core AS
        SELECT b.rid, b.route, b.horizon_min, b.snap_time, b.dest_tpl, b.pt_arr,
          b.last_actual_late_s / 60.0 AS late_now_min,
          (b.ls[-1] - b.ls[-2]) / 60.0 AS late_chg_1_min,
          (b.ls[-1] - b.ls[-4]) / 60.0 AS late_chg_3_min,
          (b.ls[-1] - b.ls[1]) / 60.0 AS late_since_start_min,
          coalesce(b.n_act, 0) AS n_actuals,
          b.fc_late_s / 60.0 AS fc_late_min,
          (b.fc_late_s - b.last_actual_late_s) / 60.0 AS fc_minus_now_min,
          CASE WHEN b.n_act > 0 THEN (epoch(b.snap_time) - b.at[-1]) / 60.0 END
            AS minutes_since_last_actual,
          r.stops_remaining,
          CASE WHEN b.n_act > 0 THEN date_diff('second', pa.plan_last, b.pt_arr) / 60.0
               ELSE date_diff('second', r.first_plan, b.pt_arr) / 60.0 END AS sched_min_remaining,
          CASE WHEN b.n_act > 0 THEN b.last_tpl ELSE r.first_tpl END AS from_tpl,
          CASE WHEN b.n_act > 0 THEN b.last_evn ELSE 'dep' END AS from_event,
          r.next_tpl, b.at[-1] AS last_actual_s
        FROM base b JOIN rem r USING (rid, route, horizon_min)
        LEFT JOIN planat pa USING (rid, route, horizon_min)
    """)
    out = _ahead(con, snaps)
    con.close()
    out["is_bank_holiday"] = float(ssd in holidays)
    local = pd.to_datetime(out["pt_arr"]).dt.tz_convert("Europe/London")
    out["arr_hour_local"] = local.dt.hour
    out["weekday"] = local.dt.dayofweek
    return out


def _empty_assoc():
    import pyarrow as pa

    return pa.table(
        {
            "main_rid": pa.array([], pa.string()),
            "assoc_rid": pa.array([], pa.string()),
            "category": pa.array([], pa.string()),
            "tiploc": pa.array([], pa.string()),
            "is_cancelled": pa.array([], pa.bool_()),
            "is_deleted": pa.array([], pa.bool_()),
            "main_wta": pa.array([], pa.timestamp("us", "UTC")),
            "assoc_wtd": pa.array([], pa.timestamp("us", "UTC")),
            "message_time": pa.array([], pa.timestamp("us", "UTC")),
        }
    )


def _ahead(con, snaps: pd.DataFrame) -> pd.DataFrame:
    w = f"INTERVAL {WINDOW_MIN} MINUTE"
    con.execute(f"""
        CREATE TEMP TABLE a_stop AS
        SELECT c.rid, c.route, c.horizon_min, count(*) AS ahead_n,
               avg(e.late_s) / 60.0 AS ahead_late_mean_min,
               avg((e.late_s >= {LATE5_S})::INT) AS ahead_late5_share
        FROM core c JOIN ev e ON e.tpl = c.from_tpl AND e.event = c.from_event AND e.pub
          AND e.rid <> c.rid AND e.message_time <= c.snap_time
          AND e.actual > c.snap_time - {w} AND epoch(e.actual) < c.last_actual_s
        WHERE c.n_actuals > 0 GROUP BY ALL
    """)
    for name, key, ev in (("dest", "dest_tpl", "arr"), ("next", "next_tpl", None)):
        evc = f"AND e.event = '{ev}'" if ev else "AND e.event = 'dep'"
        con.execute(f"""
            CREATE TEMP TABLE a_{name} AS
            SELECT c.rid, c.route, c.horizon_min, count(*) AS {name}_n,
                   avg(e.late_s) / 60.0 AS {name}_late_mean_min,
                   avg((e.late_s >= {LATE5_S})::INT) AS {name}_late5_share
            FROM core c JOIN ev e ON e.tpl = c.{key} {evc} AND e.pub AND e.rid <> c.rid
              AND e.message_time <= c.snap_time AND e.actual > c.snap_time - {w}
            GROUP BY ALL
        """)
    con.execute("""
        CREATE TEMP TABLE canc AS
        SELECT rid, tpl, min(message_time) AS first_cancel
        FROM sp WHERE cancelled GROUP BY rid, tpl
    """)
    for name, key in (("dest", "dest_tpl"), ("next", "next_tpl")):
        con.execute(f"""
            CREATE TEMP TABLE c_{name} AS
            SELECT c.rid, c.route, c.horizon_min, count(DISTINCT k.rid) AS {name}_cancel_n
            FROM core c JOIN canc k ON k.tpl = c.{key} AND k.rid <> c.rid
              AND k.first_cancel <= c.snap_time AND k.first_cancel > c.snap_time - {w}
            GROUP BY ALL
        """)
    con.execute(f"""
        CREATE TEMP TABLE net_m AS
        SELECT date_trunc('minute', message_time) AS m, count(*) AS n,
               sum((late_s >= {LATE5_S})::INT) AS n5, sum(late_s) AS sl
        FROM ev WHERE pub GROUP BY 1
    """)
    con.execute("""
        CREATE TEMP TABLE canc_m AS
        SELECT date_trunc('minute', first_cancel) AS m, count(DISTINCT rid) AS n
        FROM canc GROUP BY 1
    """)
    con.execute(f"""
        CREATE TEMP TABLE net AS
        SELECT c.rid, c.route, c.horizon_min, sum(m.n) AS net_n,
               sum(m.n5) / nullif(sum(m.n), 0) AS net_late5_share,
               sum(m.sl) / nullif(sum(m.n), 0) / 60.0 AS net_late_mean_min
        FROM core c JOIN net_m m ON m.m < date_trunc('minute', c.snap_time)
          AND m.m >= date_trunc('minute', c.snap_time) - {w}
        GROUP BY c.rid, c.route, c.horizon_min
    """)
    con.execute(f"""
        CREATE TEMP TABLE netc AS
        SELECT c.rid, c.route, c.horizon_min, sum(k.n) AS net_cancel_n
        FROM core c JOIN canc_m k ON k.m < date_trunc('minute', c.snap_time)
          AND k.m >= date_trunc('minute', c.snap_time) - {w}
        GROUP BY c.rid, c.route, c.horizon_min
    """)
    con.execute("""
        CREATE TEMP TABLE asc_t AS
        SELECT srid, horizon_min, route, snap_time, main_rid, assoc_rid, category, tiploc,
               main_wta, assoc_wtd
        FROM (
          SELECT s.rid AS srid, s.horizon_min, s.route, s.snap_time, a.main_rid, a.assoc_rid,
                 a.tiploc, a.category,
                 arg_max(a.is_cancelled, a.message_time) AS is_cancelled,
                 arg_max(a.is_deleted, a.message_time) AS is_deleted,
                 arg_max(a.main_wta, a.message_time) AS main_wta,
                 arg_max(a.assoc_wtd, a.message_time) AS assoc_wtd
          FROM s JOIN asc_raw a ON (a.assoc_rid = s.rid OR a.main_rid = s.rid)
            AND a.message_time <= s.snap_time
          GROUP BY s.rid, s.horizon_min, s.route, s.snap_time, a.main_rid, a.assoc_rid,
                   a.tiploc, a.category
        ) WHERE NOT is_cancelled AND NOT is_deleted
    """)
    con.execute("""
        CREATE TEMP TABLE asc_m AS
        SELECT x.srid, x.route, x.horizon_min, x.main_rid, x.assoc_rid, x.category, x.tiploc,
               x.main_wta, x.assoc_wtd,
               arg_max(m.late_s, {'a': m.actual, 'm': m.message_time, 't': m.tpl, 'e': m.event})
                 AS main_late_s,
               coalesce(bool_or(m.tpl = x.tiploc AND m.event = 'arr'), false) AS main_arrived
        FROM asc_t x LEFT JOIN ev m ON m.rid = x.main_rid AND m.message_time <= x.snap_time
        GROUP BY ALL
    """)
    con.execute("""
        CREATE TEMP TABLE assoc_f AS
        SELECT srid AS rid, route, horizon_min,
          count(*) FILTER (WHERE assoc_rid = srid) AS assoc_in_n,
          count(*) FILTER (WHERE main_rid = srid) AS assoc_out_n,
          count(*) FILTER (WHERE category = 'JJ') AS assoc_join_n,
          count(*) FILTER (WHERE category = 'VV') AS assoc_divide_n,
          count(*) FILTER (WHERE category = 'NP') AS assoc_next_n,
          max(main_late_s) FILTER (WHERE assoc_rid = srid) / 60.0 AS assoc_in_main_late_min,
          max(main_arrived::INT) FILTER (WHERE assoc_rid = srid) AS assoc_in_main_arrived,
          min(date_diff('second', main_wta, assoc_wtd)) FILTER (WHERE assoc_rid = srid)
              / 60.0 AS assoc_in_turn_min
        FROM asc_m GROUP BY srid, route, horizon_min
    """)
    frame = con.execute("""
        SELECT c.*, a.ahead_n, a.ahead_late_mean_min, a.ahead_late5_share,
               d.dest_n, d.dest_late_mean_min, d.dest_late5_share, cd.dest_cancel_n,
               n.next_n, n.next_late_mean_min, n.next_late5_share, cn.next_cancel_n,
               t.net_n, t.net_late5_share, t.net_late_mean_min, nc.net_cancel_n,
               f.assoc_in_n, f.assoc_in_main_late_min, f.assoc_in_main_arrived,
               f.assoc_in_turn_min, f.assoc_out_n, f.assoc_join_n, f.assoc_divide_n,
               f.assoc_next_n
        FROM core c
        LEFT JOIN a_stop a USING (rid, route, horizon_min)
        LEFT JOIN a_dest d USING (rid, route, horizon_min)
        LEFT JOIN a_next n USING (rid, route, horizon_min)
        LEFT JOIN c_dest cd USING (rid, route, horizon_min)
        LEFT JOIN c_next cn USING (rid, route, horizon_min)
        LEFT JOIN net t USING (rid, route, horizon_min)
        LEFT JOIN netc nc USING (rid, route, horizon_min)
        LEFT JOIN assoc_f f USING (rid, route, horizon_min)
    """).df()
    # counts are zero, not missing, when nothing was seen in the window
    for col in (
        "ahead_n", "dest_n", "dest_cancel_n", "next_n", "next_cancel_n", "net_n", "net_cancel_n",
        "assoc_in_n", "assoc_out_n", "assoc_join_n", "assoc_divide_n", "assoc_next_n",
    ):  # fmt: skip
        frame[col] = frame[col].fillna(0)
    return frame


def fastest_table(
    root: Path,
    days: list[date],
    pairs: pd.DataFrame,
    min_obs: int = MIN_PAIR_OBS,
    exclude_rids: set[str] | None = None,
) -> pd.DataFrame:
    """5th percentile of the observed run time from an event at a stop to the arrival at the
    destination, over the given (train) days, for the (from_tpl, from_event, dest_tpl) pairs.
    `exclude_rids` drops those services (cold start: the held-out routes' own services)."""
    import duckdb

    con = duckdb.connect()
    con.register("p", pairs[["from_tpl", "from_event", "dest_tpl"]].drop_duplicates())
    con.register("x", pd.DataFrame({"rid": sorted(exclude_rids or set())}, dtype="string"))
    parts = []
    for d in days:
        a = train_pool_table(root, "stop_actuals", ["rid", "tpl", "event", "actual"], [d])
        if a.num_rows == 0:
            continue
        con.register("a", a)
        parts.append(
            con.execute("""
                SELECT p.from_tpl, p.from_event, p.dest_tpl,
                       date_diff('second', f.actual, t.actual) AS run_s
                FROM p JOIN a f ON f.tpl = p.from_tpl AND f.event = p.from_event
                JOIN a t ON t.rid = f.rid AND t.tpl = p.dest_tpl AND t.event = 'arr'
                WHERE f.actual IS NOT NULL AND t.actual IS NOT NULL AND t.actual > f.actual
                  AND f.rid NOT IN (SELECT rid FROM x)
            """).df()
        )
        con.unregister("a")
    con.close()
    if not parts:
        return pd.DataFrame(columns=[*INTERNAL, "fastest_min", "n_obs"])
    runs = pd.concat(parts, ignore_index=True)
    g = runs.groupby(INTERNAL)["run_s"]
    out = g.quantile(0.05).div(60.0).rename("fastest_min").to_frame().join(g.size().rename("n_obs"))
    out = out.reset_index()
    return out[out["n_obs"] >= min_obs].reset_index(drop=True)


def add_slack(feats: pd.DataFrame, fastest: pd.DataFrame) -> pd.DataFrame:
    """fastest_min_remaining, slack_min = scheduled time left minus the fastest observed, and the
    lateness left after using all slack. NaN when the pair was seen too rarely in train dates."""
    out = feats.merge(fastest[[*INTERNAL, "fastest_min"]], on=INTERNAL, how="left")
    out["fastest_min_remaining"] = out["fastest_min"]
    out["slack_min"] = out["sched_min_remaining"] - out["fastest_min"]
    out["late_minus_slack_min"] = out["late_now_min"].fillna(0.0) - out["slack_min"].clip(lower=0)
    return out.drop(columns="fastest_min")


def assert_clean_feature_names(names: list[str]) -> None:
    bad = [n for n in names if any(b in n.lower() for b in BANNED_SUBSTRINGS)]
    if bad:
        raise ValueError(f"banned identity columns reached the model: {bad}")


def as_matrix(df: pd.DataFrame) -> pd.DataFrame:
    assert_clean_feature_names(FEATURES)
    return df[FEATURES].astype("float64").replace([np.inf, -np.inf], np.nan)


def build_range(
    root: Path, snap_dir: Path, out_dir: Path, days: list[date], rebuild: bool = False
) -> dict:
    """One Parquet per service date (features plus the snapshot columns the scoring needs)."""
    holidays = bank_holidays()
    report: dict = {"built": 0, "skipped_existing": 0, "missing_days": [], "rows": 0}
    keep = [*KEY, "ssd", "snap_time", "pt_arr", "toc", "target_late_s", "has_target"]
    for d in days:
        dest = out_dir / f"service_date={d}" / "part.parquet"
        src = snap_dir / f"service_date={d}" / "part.parquet"
        if dest.exists() and not rebuild:
            report["skipped_existing"] += 1
            continue
        if not src.exists():
            report["missing_days"].append(str(d))
            continue
        snaps = pd.read_parquet(src)
        feats = build_day(root, d, snaps, holidays)
        if feats.empty:
            report["missing_days"].append(str(d))
            continue
        extra = snaps[[c for c in keep if c in snaps and c not in feats]]
        keys = snaps[KEY + [c for c in extra if c not in KEY]]
        out = feats.merge(keys, on=KEY, how="left")
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dest, index=False)
        report["built"] += 1
        report["rows"] += len(out)
    return report
