"""State of a service at snapshot time T, built only from messages with message_time <= T.

One definition, written for three engines (DuckDB, Polars, Spark) and a plain pandas reference
used by the leakage check. The state is:
  last_actual_*: the most advanced actual (latest actual time, then latest message, stop, event)
                 reported by T, and its lateness against the public time (else working time);
  fc_late_s:     the latest Darwin forecast arrival at the destination by T, minus the plan.
Lateness is in whole seconds, which keeps the engines' outputs exactly comparable.
"""

import pandas as pd
import pyarrow as pa

UPDATE_COLS = [
    "rid",
    "tpl",
    "event",
    "pt_arr",
    "pt_dep",
    "wt_arr",
    "wt_dep",
    "wt_pass",
    "est",
    "actual",
    "message_time",
]
KEYS = ["rid", "route", "horizon_min"]
STATE_COLS = [*KEYS, "last_actual_tpl", "last_actual_s", "last_actual_late_s", "fc_late_s"]
SNAP_COLS = [*KEYS, "snap_time", "dest_tpl", "pt_arr"]


def _snaps_arrow(snaps: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(snaps[SNAP_COLS], preserve_index=False)


def state_duckdb(updates: pa.Table, snaps: pd.DataFrame) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("u", updates)
    con.register("s", _snaps_arrow(snaps))
    con.execute(
        """
        CREATE TEMP TABLE a AS
        SELECT rid, tpl, event, message_time, actual,
               date_diff('second',
                 CASE event WHEN 'arr' THEN coalesce(pt_arr, wt_arr)
                            WHEN 'dep' THEN coalesce(pt_dep, wt_dep) ELSE wt_pass END,
                 actual) AS late_s
        FROM u WHERE actual IS NOT NULL
          AND CASE event WHEN 'arr' THEN coalesce(pt_arr, wt_arr)
                         WHEN 'dep' THEN coalesce(pt_dep, wt_dep) ELSE wt_pass END IS NOT NULL
        """
    )
    last = con.execute(
        """
        SELECT s.rid, s.route, s.horizon_min,
          arg_max({'tpl': a.tpl, 'sec': epoch(a.actual)::BIGINT, 'late': a.late_s},
                  {'t': a.actual, 'm': a.message_time, 'p': a.tpl, 'e': a.event}) AS x
        FROM s JOIN a ON a.rid = s.rid AND a.message_time <= s.snap_time
        GROUP BY s.rid, s.route, s.horizon_min
        """
    ).df()
    fc = con.execute(
        """
        SELECT s.rid, s.route, s.horizon_min,
          date_diff('second', any_value(s.pt_arr),
                    arg_max(f.est, {'m': f.message_time, 'e': f.est})) AS fc_late_s
        FROM s JOIN (SELECT rid, tpl, message_time, est FROM u
                     WHERE event = 'arr' AND est IS NOT NULL) f
          ON f.rid = s.rid AND f.tpl = s.dest_tpl AND f.message_time <= s.snap_time
        GROUP BY s.rid, s.route, s.horizon_min
        """
    ).df()
    con.close()
    return _assemble(
        snaps,
        pd.DataFrame(
            {
                **{k: last[k] for k in KEYS},
                "last_actual_tpl": [v["tpl"] for v in last["x"]],
                "last_actual_s": [v["sec"] for v in last["x"]],
                "last_actual_late_s": [v["late"] for v in last["x"]],
            }
        ),
        fc,
    )


def state_polars(updates: pa.Table, snaps: pd.DataFrame) -> pd.DataFrame:
    import polars as pl

    u = pl.from_arrow(updates)
    s = pl.from_arrow(_snaps_arrow(snaps))
    planned = (
        pl.when(pl.col("event") == "arr")
        .then(pl.coalesce("pt_arr", "wt_arr"))
        .when(pl.col("event") == "dep")
        .then(pl.coalesce("pt_dep", "wt_dep"))
        .otherwise(pl.col("wt_pass"))
    )
    a = (
        u.filter(pl.col("actual").is_not_null())
        .with_columns(planned.alias("plan"))
        .filter(pl.col("plan").is_not_null())
        .select(
            "rid",
            "tpl",
            "event",
            "message_time",
            "actual",
            ((pl.col("actual") - pl.col("plan")).dt.total_seconds()).alias("late_s"),
        )
    )
    last = (
        s.join(a, on="rid")
        .filter(pl.col("message_time") <= pl.col("snap_time"))
        .sort([*KEYS, "actual", "message_time", "tpl", "event"])
        .group_by(KEYS, maintain_order=True)
        .agg(
            pl.col("tpl").last().alias("last_actual_tpl"),
            pl.col("actual").last().dt.epoch("s").alias("last_actual_s"),
            pl.col("late_s").last().alias("last_actual_late_s"),
        )
        .to_pandas()
    )
    f = u.filter((pl.col("event") == "arr") & pl.col("est").is_not_null()).select(
        "rid", "tpl", "message_time", "est"
    )
    fc = (
        s.join(f, left_on=["rid", "dest_tpl"], right_on=["rid", "tpl"])
        .filter(pl.col("message_time") <= pl.col("snap_time"))
        .sort([*KEYS, "message_time", "est"])
        .group_by(KEYS, maintain_order=True)
        .agg(
            ((pl.col("est").last() - pl.col("pt_arr").first()).dt.total_seconds()).alias(
                "fc_late_s"
            )
        )
        .to_pandas()
    )
    return _assemble(snaps, last, fc)


def state_spark(updates: pa.Table, snaps: pd.DataFrame) -> pd.DataFrame:
    from pyspark.sql import functions as F

    from rail_risk.spark import build_spark

    spark = build_spark("state-engine-bench", driver_memory="8g")
    u = spark.createDataFrame(updates.to_pandas())
    s = spark.createDataFrame(snaps[SNAP_COLS])
    plan = (
        F.when(F.col("event") == "arr", F.coalesce("pt_arr", "wt_arr"))
        .when(F.col("event") == "dep", F.coalesce("pt_dep", "wt_dep"))
        .otherwise(F.col("wt_pass"))
    )
    a = (
        u.where(F.col("actual").isNotNull())
        .withColumn("plan", plan)
        .where(F.col("plan").isNotNull())
        .withColumn("late_s", F.col("actual").cast("long") - F.col("plan").cast("long"))
    )
    order = F.struct("actual", "message_time", "tpl", "event")
    last = (
        s.join(a, "rid")
        .where(F.col("message_time") <= F.col("snap_time"))
        .groupBy(*KEYS)
        .agg(
            F.max_by(
                F.struct(F.col("tpl"), F.col("actual").cast("long").alias("sec"), "late_s"), order
            ).alias("x")
        )
        .select(
            *KEYS,
            F.col("x.tpl").alias("last_actual_tpl"),
            F.col("x.sec").alias("last_actual_s"),
            F.col("x.late_s").alias("last_actual_late_s"),
        )
        .toPandas()
    )
    f = u.where((F.col("event") == "arr") & F.col("est").isNotNull()).select(
        "rid", F.col("tpl").alias("ftpl"), "message_time", "est"
    )
    fc = (
        s.join(f, (s.rid == f.rid) & (s.dest_tpl == f.ftpl))
        .drop(f.rid)
        .where(F.col("message_time") <= F.col("snap_time"))
        .groupBy(*KEYS)
        .agg(
            F.max_by(F.col("est"), F.struct("message_time", "est")).alias("e"),
            F.first("pt_arr").alias("p"),
        )
        .select(*KEYS, (F.col("e").cast("long") - F.col("p").cast("long")).alias("fc_late_s"))
        .toPandas()
    )
    return _assemble(snaps, last, fc)


def state_reference(updates: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """The same state by a plain loop: slow, obviously correct, used to check the engines."""
    rows = []
    by_rid = {rid: g for rid, g in updates.groupby("rid")}
    for snap in snaps.itertuples(index=False):
        g = by_rid.get(snap.rid)
        out = dict(rid=snap.rid, route=snap.route, horizon_min=snap.horizon_min)
        if g is not None:
            g = g[g["message_time"] <= snap.snap_time]
            a = g[g["actual"].notna()].copy()
            plan = a["wt_pass"].copy()
            arr = a["event"] == "arr"
            dep = a["event"] == "dep"
            plan[arr] = a["pt_arr"].fillna(a["wt_arr"])[arr]
            plan[dep] = a["pt_dep"].fillna(a["wt_dep"])[dep]
            a = a.assign(plan=plan)
            a = a[a["plan"].notna()]
            if len(a):
                top = a.sort_values(["actual", "message_time", "tpl", "event"]).iloc[-1]
                out["last_actual_tpl"] = top["tpl"]
                out["last_actual_s"] = int(top["actual"].timestamp())
                out["last_actual_late_s"] = int((top["actual"] - top["plan"]).total_seconds())
            f = g[(g["event"] == "arr") & (g["tpl"] == snap.dest_tpl) & g["est"].notna()]
            if len(f):
                best = f.sort_values(["message_time", "est"]).iloc[-1]
                out["fc_late_s"] = int((best["est"] - snap.pt_arr).total_seconds())
        rows.append(out)
    return _assemble(
        snaps, pd.DataFrame(rows)[[c for c in STATE_COLS if c in rows[0]]] if rows else None, None
    )


ENGINES = {"duckdb": state_duckdb, "polars": state_polars, "spark": state_spark}


def _assemble(
    snaps: pd.DataFrame, last: pd.DataFrame | None, fc: pd.DataFrame | None
) -> pd.DataFrame:
    """Every snapshot row, with the state columns filled where messages existed by T."""
    out = snaps[KEYS].copy()
    for part in (last, fc):
        if part is not None and len(part):
            out = out.merge(part, on=KEYS, how="left", suffixes=("", "_dup"))
            out = out[[c for c in out.columns if not c.endswith("_dup")]]
    for c in STATE_COLS:
        if c not in out:
            out[c] = pd.NA
    out = out[STATE_COLS].sort_values(KEYS).reset_index(drop=True)
    for c in ("last_actual_s", "last_actual_late_s", "fc_late_s"):
        out[c] = out[c].astype("Int64")
    out["last_actual_tpl"] = out["last_actual_tpl"].astype("string")
    return out
