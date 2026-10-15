"""Point-in-time feature snapshots from HSP stop data.

One row per service (rid), route and prediction horizon. The prediction time T is the scheduled
destination arrival minus the horizon. Live-state features use only events whose actual time is
at or before T. Historical features use only services dated strictly before the service date.
The label (arrival delay at the route destination) and the candidate-cancellation flag are
outcomes and are never used as features.
"""

import json
import sqlite3
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window

HORIZONS = (120, 60, 15)
HISTORY_DAYS = 28

BANK_HOLIDAYS_FILE = Path(__file__).resolve().parents[3] / "config" / "bank_holidays.json"


def load_bank_holidays(path: Path = BANK_HOLIDAYS_FILE) -> tuple[str, ...]:
    """England and Wales bank holiday dates, from the file fetched from gov.uk (see the file)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(e["date"] for e in data["events"])


KEYS = ["rid", "route_id", "horizon_min"]
LIVE_FEATURES = [
    "last_stop_seq",
    "last_stop_lateness_min",
    "minutes_since_last_stop",
    "stops_remaining",
    "sched_minutes_remaining",
    "lateness_change_last_2",
    "n_stops_passed",
]
CONTEXT_FEATURES = [
    "hour_of_day",
    "dest_hour",
    "day_of_week",
    "is_bank_holiday",
    "toc_code",
]
HISTORY_FEATURES = ["hist_median_28d", "hist_p80_28d", "hist_n_28d"]
FEATURE_COLUMNS = LIVE_FEATURES + CONTEXT_FEATURES + HISTORY_FEATURES
OUTCOME_COLUMNS = ["label_delay_min", "is_candidate_cancelled"]

STOPS_SCHEMA = pa.schema(
    [
        ("rid", pa.string()),
        ("date_of_service", pa.date32()),
        ("toc_code", pa.string()),
        ("seq", pa.int64()),
        ("location", pa.string()),
        ("sched_arr", pa.timestamp("us", tz="UTC")),
        ("sched_dep", pa.timestamp("us", tz="UTC")),
        ("actual_arr", pa.timestamp("us", tz="UTC")),
        ("actual_dep", pa.timestamp("us", tz="UTC")),
        ("late_canc_reason", pa.string()),
        ("candidate_cancelled", pa.bool_()),
    ]
)


def unify_stops(src: Path, dst: Path) -> int:
    """Combine the interim HSP stop files into one file with one stable schema.

    The files are written one window at a time by pandas, so a window with no reason codes has a
    null-typed reason column and timestamps are nanoseconds, neither of which Spark reads as one
    table. The data is small, so it is normalised through Arrow. Returns the row count.
    """
    tables = []
    for path in sorted(src.glob("date=*/*.parquet")):
        try:
            table = pq.read_table(path)
        except pa.ArrowInvalid:  # a file still being written by an older writer
            continue
        columns = {}
        for field in STOPS_SCHEMA:
            col = table.column(field.name)
            if field.name == "date_of_service":
                col = col.cast(pa.timestamp("us")).cast(pa.date32())
            elif pa.types.is_timestamp(field.type):
                col = col.cast(pa.timestamp("us", tz="UTC"))
            else:
                col = col.cast(field.type)
            columns[field.name] = col
        tables.append(pa.table(columns, schema=STOPS_SCHEMA))
    dst.parent.mkdir(parents=True, exist_ok=True)
    merged = pa.concat_tables(tables) if tables else STOPS_SCHEMA.empty_table()
    pq.write_table(merged, dst)
    return merged.num_rows


def load_stops(spark: SparkSession, path: Path) -> DataFrame:
    """The unified HSP stops file (see unify_stops)."""
    return (
        spark.read.parquet(str(path))
        .withColumnRenamed("date_of_service", "service_date")
        .dropDuplicates(["rid", "seq"])
    )


def load_rid_routes(spark: SparkSession, manifest: Path) -> DataFrame:
    con = sqlite3.connect(f"file:{manifest}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT rid, route FROM rid_routes").fetchall()
    finally:
        con.close()
    return spark.createDataFrame(rows, "rid string, route_id string")


def _minutes(later: F.Column, earlier: F.Column) -> F.Column:
    return (later.cast("long") - earlier.cast("long")) / 60.0


def destination_services(
    spark: SparkSession, stops: DataFrame, rid_routes: DataFrame, route_dest: dict[str, str]
) -> tuple[DataFrame, DataFrame]:
    """(services, stop_rows): one service row per (rid, route) at its route destination, and the
    stop rows up to and including that destination."""
    dest = spark.createDataFrame(list(route_dest.items()), "route_id string, dest_crs string")
    s = stops.join(rid_routes, "rid").join(dest, "route_id")
    pick = Window.partitionBy("rid", "route_id").orderBy(F.col("seq").desc())
    services = (
        s.filter((F.col("location") == F.col("dest_crs")) & F.col("sched_arr").isNotNull())
        .withColumn("_rn", F.row_number().over(pick))
        .filter("_rn = 1")
        .select(
            "rid",
            "route_id",
            "service_date",
            "toc_code",
            "candidate_cancelled",
            F.col("seq").alias("dest_seq"),
            F.col("sched_arr").alias("dest_sched_arr"),
            F.col("actual_arr").alias("dest_actual_arr"),
        )
    )
    rows = s.join(services.select("rid", "route_id", "dest_seq"), ["rid", "route_id"]).filter(
        F.col("seq") <= F.col("dest_seq")
    )
    return services, rows


def history_labels(services: DataFrame) -> DataFrame:
    """One completed-delay row per service, the only input to the historical features."""
    return (
        services.filter(~F.col("candidate_cancelled") & F.col("dest_actual_arr").isNotNull())
        .withColumn("dest_hour", F.hour(F.from_utc_timestamp("dest_sched_arr", "Europe/London")))
        .select(
            "route_id",
            "service_date",
            "dest_hour",
            _minutes(F.col("dest_actual_arr"), F.col("dest_sched_arr")).alias("delay_min"),
        )
    )


def build_snapshots(
    spark: SparkSession,
    stops: DataFrame,
    rid_routes: DataFrame,
    route_dest: dict[str, str],
    horizons: tuple[int, ...] = HORIZONS,
    labels: DataFrame | None = None,
    bank_holidays: tuple[str, ...] | None = None,
) -> DataFrame:
    """Feature snapshots. `labels` overrides the source of the historical features, so a subset
    of services can be rebuilt against the full history."""
    holidays = bank_holidays if bank_holidays is not None else load_bank_holidays()
    services, rows = destination_services(spark, stops, rid_routes, route_dest)
    hz = spark.createDataFrame([(h,) for h in horizons], "horizon_min int")
    base = services.crossJoin(hz).withColumn(
        "t", (F.col("dest_sched_arr").cast("long") - F.col("horizon_min") * 60).cast("timestamp")
    )

    # observed events (arrival or departure with an actual time) at or before T
    arr = rows.select(
        "rid", "route_id", "seq", F.lit(0).alias("kind"), F.col("sched_arr").alias("sched"),
        F.col("actual_arr").alias("actual"), "late_canc_reason",
    )  # fmt: skip
    dep = rows.select(
        "rid", "route_id", "seq", F.lit(1).alias("kind"), F.col("sched_dep").alias("sched"),
        F.col("actual_dep").alias("actual"), "late_canc_reason",
    )  # fmt: skip
    events = (
        arr.unionByName(dep)
        .filter(F.col("sched").isNotNull() & F.col("actual").isNotNull())
        .join(base.select("rid", "route_id", "horizon_min", "t"), ["rid", "route_id"])
        .filter(F.col("actual") <= F.col("t"))
    )
    per_stop = Window.partitionBy("rid", "route_id", "horizon_min", "seq").orderBy(
        F.col("actual").desc(), F.col("kind").desc()
    )
    stops_seen = events.withColumn("_k", F.row_number().over(per_stop)).filter("_k = 1")
    by_seq = Window.partitionBy("rid", "route_id", "horizon_min").orderBy(F.col("seq").desc())
    ranked = stops_seen.withColumn("_rn", F.row_number().over(by_seq)).withColumn(
        "lateness", _minutes(F.col("actual"), F.col("sched"))
    )
    totals = stops_seen.groupBy("rid", "route_id", "horizon_min").agg(
        F.count("*").alias("n_stops_passed")
    )
    last = ranked.filter("_rn = 1").select(
        "rid", "route_id", "horizon_min",
        F.col("seq").alias("last_stop_seq"), F.col("actual").alias("_last_actual"),
        F.col("sched").alias("_last_sched"), F.col("lateness").alias("last_stop_lateness_min"),
    )  # fmt: skip
    prev = ranked.filter("_rn = 2").select(
        "rid", "route_id", "horizon_min", F.col("lateness").alias("_prev_lateness")
    )

    local_t = F.from_utc_timestamp("t", "Europe/London")
    snap = (
        base.join(last, KEYS, "left")
        .join(prev, KEYS, "left")
        .join(totals, KEYS, "left")
        .withColumn("minutes_since_last_stop", _minutes(F.col("t"), F.col("_last_actual")))
        .withColumn("stops_remaining", F.col("dest_seq") - F.col("last_stop_seq"))
        .withColumn(
            "sched_minutes_remaining", _minutes(F.col("dest_sched_arr"), F.col("_last_sched"))
        )
        .withColumn(
            "lateness_change_last_2", F.col("last_stop_lateness_min") - F.col("_prev_lateness")
        )
        .withColumn("hour_of_day", F.hour(local_t))
        .withColumn("dest_hour", F.hour(F.from_utc_timestamp("dest_sched_arr", "Europe/London")))
        .withColumn("day_of_week", ((F.dayofweek("service_date") + 5) % 7) + 1)
        .withColumn("is_bank_holiday", F.col("service_date").cast("string").isin(*holidays))
        .withColumn("is_candidate_cancelled", F.col("candidate_cancelled"))
        .withColumn(
            "label_delay_min",
            F.when(
                ~F.col("candidate_cancelled") & F.col("dest_actual_arr").isNotNull(),
                _minutes(F.col("dest_actual_arr"), F.col("dest_sched_arr")),
            ),
        )
    )

    # historical features: services dated 1 to 28 days before, same route and destination hour
    hist_src = (labels if labels is not None else history_labels(services)).select(
        F.col("route_id").alias("h_route"),
        F.col("service_date").alias("h_date"),
        F.col("dest_hour").alias("h_hour"),
        "delay_min",
    )
    keys = snap.select(
        F.col("route_id").alias("k_route"),
        F.col("service_date").alias("k_date"),
        F.col("dest_hour").alias("k_hour"),
    ).distinct()
    window_join = (
        (F.col("k_route") == F.col("h_route"))
        & (F.col("k_hour") == F.col("h_hour"))
        & (F.col("h_date") >= F.date_sub(F.col("k_date"), HISTORY_DAYS))
        & (F.col("h_date") < F.col("k_date"))
    )
    hist = (
        keys.join(hist_src, window_join)
        .groupBy("k_route", "k_date", "k_hour")
        .agg(
            F.percentile("delay_min", 0.5).alias("hist_median_28d"),
            F.percentile("delay_min", 0.8).alias("hist_p80_28d"),
            F.count("*").alias("hist_n_28d"),
        )
        .withColumnsRenamed(
            {"k_route": "route_id", "k_date": "service_date", "k_hour": "dest_hour"}
        )
    )
    snap = snap.join(hist, ["route_id", "service_date", "dest_hour"], "left").withColumn(
        "hist_n_28d", F.coalesce(F.col("hist_n_28d"), F.lit(0))
    )
    return snap.select(
        "rid", "route_id", "service_date", "horizon_min", "t", "dest_sched_arr",
        *FEATURE_COLUMNS, *OUTCOME_COLUMNS,
    )  # fmt: skip


def write_snapshots(snap: DataFrame, out: Path) -> None:
    (
        snap.repartition("service_date")
        .write.mode("overwrite")
        .partitionBy("service_date")
        .parquet(str(out))
    )


def route_destinations(routes_file: Path) -> dict[str, str]:
    from rail_risk.config import load_routes

    return {r.id: r.destination for r in load_routes(routes_file).routes}


def service_dates(stops: DataFrame) -> list[date]:
    return sorted(r[0] for r in stops.select("service_date").distinct().collect())
