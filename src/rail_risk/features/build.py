"""Driver for the snapshot build and the real-data leakage check."""

import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from rail_risk.features.snapshots import (
    FEATURE_COLUMNS,
    HORIZONS,
    build_snapshots,
    destination_services,
    history_labels,
    load_rid_routes,
    load_stops,
    route_destinations,
    unify_stops,
    write_snapshots,
)
from rail_risk.logging import get_logger
from rail_risk.spark import PeakMemory


@dataclass
class BuildStats:
    seconds: float
    peak_mb: float
    rows: int
    rows_per_horizon: dict[int, int]
    label_null_rate: float
    candidate_cancelled_rate: float
    services: int
    days: int


def load_inputs(
    spark: SparkSession,
    interim: Path,
    routes_file: Path,
    start: date | None = None,
    end: date | None = None,
) -> tuple[DataFrame, DataFrame, dict[str, str]]:
    unified = interim / "_unified" / "hsp_stops.parquet"
    unify_stops(interim / "hsp_stops", unified)
    stops = load_stops(spark, unified)
    if start is not None:
        stops = stops.filter(F.col("service_date") >= F.lit(start))
    if end is not None:
        stops = stops.filter(F.col("service_date") <= F.lit(end))
    rid_routes = load_rid_routes(spark, interim / "hsp_manifest.sqlite")
    return stops, rid_routes, route_destinations(routes_file)


def build(
    spark: SparkSession,
    interim: Path,
    features_root: Path,
    routes_file: Path,
    start: date | None = None,
    end: date | None = None,
) -> BuildStats:
    import time

    log = get_logger("features")
    started = time.monotonic()
    with PeakMemory(spark) as mem:
        stops, rid_routes, route_dest = load_inputs(spark, interim, routes_file, start, end)
        snap = build_snapshots(spark, stops, rid_routes, route_dest)
        out = features_root / "snapshots"
        write_snapshots(snap, out)
        written = spark.read.parquet(str(out))
        agg = written.agg(
            F.count("*").alias("rows"),
            F.avg(F.col("label_delay_min").isNull().cast("double")).alias("label_null"),
            F.avg(F.col("is_candidate_cancelled").cast("double")).alias("cancelled"),
            F.countDistinct("rid", "route_id").alias("services"),
            F.countDistinct("service_date").alias("days"),
        ).first()
        per_h = {
            r["horizon_min"]: r["n"]
            for r in written.groupBy("horizon_min").agg(F.count("*").alias("n")).collect()
        }
    stats = BuildStats(
        seconds=time.monotonic() - started,
        peak_mb=mem.peak_mb,
        rows=agg["rows"],
        rows_per_horizon=dict(sorted(per_h.items())),
        label_null_rate=agg["label_null"],
        candidate_cancelled_rate=agg["cancelled"],
        services=agg["services"],
        days=agg["days"],
    )
    log.info("snapshots_built", **stats.__dict__)
    return stats


def leakage_check_real(
    spark: SparkSession,
    interim: Path,
    routes_file: Path,
    n: int = 50,
    seed: int = 7,
    start: date | None = None,
    end: date | None = None,
    routes: tuple[str, ...] | None = None,
    last_days: int | None = None,
) -> tuple[int, int, list[dict], dict[str, int]]:
    """Rebuild `n` random real rows after deleting every actual time after T, and compare.

    The sample is split evenly across `routes` (every route present when None), so each route
    is checked and not just the one with the most rows. `last_days` keeps only the newest
    service dates and `routes` only those routes, which bounds memory in a small test session.

    Each sampled (service, route, horizon) becomes its own pseudo service whose stops have all
    actuals after that row's T set to null. The rebuild uses the full history labels, so only
    the live-state and context features can differ. Returns (rows checked, sampled services that
    had future actuals deleted, mismatches, rows checked per route).
    """
    stops, rid_routes, route_dest = load_inputs(spark, interim, routes_file, start, end)
    if last_days is not None:
        newest = stops.agg(F.max("service_date")).first()[0]
        stops = stops.filter(F.col("service_date") > F.date_sub(F.lit(newest), last_days))
    if routes is not None:
        rid_routes = rid_routes.filter(F.col("route_id").isin(list(routes)))
    full = build_snapshots(spark, stops, rid_routes, route_dest).cache()
    rows = full.select("rid", "route_id", "horizon_min", "t", *FEATURE_COLUMNS).collect()
    rng = random.Random(seed)  # noqa: S311 - sampling, not security
    present = sorted({r["route_id"] for r in rows})
    chosen = [r for r in present if routes is None or r in routes]
    sample = []
    for route in chosen:
        mine = [r for r in rows if r["route_id"] == route]
        rng.shuffle(mine)
        sample += mine[: n // len(chosen)]
    per_route = {route: sum(1 for r in sample if r["route_id"] == route) for route in chosen}

    services, _ = destination_services(spark, stops, rid_routes, route_dest)
    labels = history_labels(services)

    picked = spark.createDataFrame(
        [(r["rid"], r["route_id"], r["t"], i) for i, r in enumerate(sample)],
        "rid string, route_id string, t timestamp, idx int",
    )
    pseudo = F.concat(F.col("rid"), F.lit("#"), F.col("idx").cast("string"))
    columns = stops.columns
    joined = stops.join(picked.select("rid", "t", "idx"), "rid")
    future = (F.col("actual_arr") > F.col("t")) | (F.col("actual_dep") > F.col("t"))
    deleted = joined.filter(future).select("rid", "idx").distinct().count()
    alt_stops = joined
    for col in ("actual_arr", "actual_dep"):
        alt_stops = alt_stops.withColumn(
            col, F.when(F.col(col) > F.col("t"), None).otherwise(F.col(col))
        )
    alt_stops = alt_stops.withColumn("rid", pseudo).select(*columns)
    alt_routes = picked.select(pseudo.alias("rid"), "route_id")
    rebuilt = build_snapshots(spark, alt_stops, alt_routes, route_dest, labels=labels)
    got = {
        (r["rid"], r["route_id"], r["horizon_min"]): r
        for r in rebuilt.select("rid", "route_id", "horizon_min", *FEATURE_COLUMNS).collect()
    }
    mismatches = []
    for i, r in enumerate(sample):
        other = got.get((f"{r['rid']}#{i}", r["route_id"], r["horizon_min"]))
        for col in FEATURE_COLUMNS:
            a, b = r[col], other[col] if other else "missing row"
            same = (a is None and b is None) or (
                a is not None
                and b is not None
                and (a == b or (isinstance(a, float) and abs(a - b) < 1e-9))
            )
            if not same:
                mismatches.append(
                    {"rid": r["rid"], "horizon": r["horizon_min"], "feature": col, "a": a, "b": b}
                )
    full.unpersist()
    return len(sample), deleted, mismatches, per_route


__all__ = ["HORIZONS", "BuildStats", "build", "leakage_check_real", "load_inputs"]
