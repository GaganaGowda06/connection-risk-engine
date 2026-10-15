"""Synthetic data for tests and for the container smoke test in CI.

Nothing here is real. `build_fixture_store` writes a tiny `data/` and `mlruns/` that the API can
start from, with a registered champion model, so an image can be checked without real data.
"""

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rail_risk.evaluation import Bands, HistoricalBaseline, SplitConfig, make_split
from rail_risk.features.snapshots import FEATURE_COLUMNS, STOPS_SCHEMA
from rail_risk.models.bundle import ModelBundle, save_bundle
from rail_risk.models.pipeline import fit_all

BANDS = Bands((0, 15, 30, 60, 120), provisional=True)
CFG = {
    "quantiles": [0.1, 0.5, 0.9],
    "nominal_coverage": 0.8,
    "split_validation_min_days": 10,
    "seed": 11,
    "num_threads": 1,
    "max_rounds": 60,
    "early_stopping_rounds": 10,
    "grid": {
        "num_leaves": [4],
        "min_data_in_leaf": [20],
        "learning_rate": [0.1],
        "feature_fraction": [0.8],
    },
    "fixed": {"lambda_l2": 10.0, "max_depth": 5, "bagging_fraction": 0.8, "bagging_freq": 1},
    "band_model": {
        "num_leaves": 4,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
    },  # fmt: skip
}


def snapshots(days: int = 26, per_day: int = 24, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2026, 8, 1)
    for i in range(days):
        d = start + timedelta(days=i)
        for k in range(per_day):
            base = rng.gamma(2.0, 3.0)
            for h in (120, 60, 15):
                row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
                row.update(
                    {
                        "rid": f"{d:%Y%m%d}{k:07d}",
                        "route_id": "bhm-eus",
                        "service_date": d,
                        "horizon_min": h,
                        "toc_code": "VT" if k % 2 else "LM",
                        "dest_hour": 9 + k % 3,
                        "is_bank_holiday": False,
                        "last_stop_lateness_min": None if h == 120 else base * 0.7,
                        "hist_median_28d": None if i < 2 else 3.0,
                        "label_delay_min": base,
                        "is_candidate_cancelled": False,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def build_bundle() -> tuple[ModelBundle, HistoricalBaseline]:
    df = snapshots()
    split = make_split(sorted(df["service_date"].unique()), SplitConfig(7, 7, 1, 16, 50, 1, 0.95))
    block = {d: n for n, days in split.blocks().items() for d in days}
    df = df[df["service_date"].isin(block)].assign(block=lambda x: x["service_date"].map(block))
    fitted = fit_all(df[df["block"] == "train"], df[df["block"] == "validation"], BANDS, CFG)
    bundle = ModelBundle(fitted.quantile, fitted.bands, BANDS, meta={"git_commit": "fixture"})
    return bundle, fitted.baseline


def build_fixture_store(root: Path, days: int = 8) -> str:
    """Write root/data and root/mlruns. Returns the registered run id."""
    import mlflow

    from rail_risk.models.registry import MODEL_NAME, register_champion

    interim = root / "data" / "interim"
    (interim / "_unified").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw" / "darwin").mkdir(parents=True, exist_ok=True)

    # eight earlier days of bhm-eus services that reach EUS, for the 28-day history features
    rows, rid_routes = [], []
    today = datetime.now(UTC).date()
    for k in range(1, days + 1):
        day = today - timedelta(days=k)
        for j in range(4):
            rid = f"{day:%Y%m%d}{9000000 + j:07d}"
            arr = datetime(day.year, day.month, day.day, 10 + j, 30, tzinfo=UTC)
            late = timedelta(minutes=2 + (j + k) % 6)
            for seq, (loc, sa, sd) in enumerate(
                [("BHM", None, arr - timedelta(hours=1, minutes=30)), ("EUS", arr, None)]
            ):
                rows.append(
                    {
                        "rid": rid, "date_of_service": day, "toc_code": "VT", "seq": seq,
                        "location": loc, "sched_arr": sa, "sched_dep": sd,
                        "actual_arr": (sa + late) if sa else None,
                        "actual_dep": (sd + timedelta(minutes=1)) if sd else None,
                        "late_canc_reason": None, "candidate_cancelled": False,
                    }
                )  # fmt: skip
            rid_routes.append((rid, "bhm-eus"))
    table = pa.Table.from_pandas(pd.DataFrame(rows), schema=STOPS_SCHEMA, preserve_index=False)
    pq.write_table(table, interim / "_unified" / "hsp_stops.parquet")
    db = sqlite3.connect(interim / "hsp_manifest.sqlite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS rid_routes (rid TEXT, route TEXT, PRIMARY KEY (rid, route))"
    )
    db.executemany("INSERT OR REPLACE INTO rid_routes VALUES (?, ?)", rid_routes)
    db.commit()
    db.close()

    tracking = root / "mlruns"
    tracking.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(tracking / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if mlflow.get_experiment_by_name("fixture") is None:
        mlflow.create_experiment(
            "fixture", artifact_location=(tracking / "artifacts").resolve().as_uri()
        )
    mlflow.set_experiment("fixture")
    bundle, baseline = build_bundle()
    with mlflow.start_run(run_name="fixture-model") as run:
        mlflow.set_tags({"stage": "reportable", "git_commit": "fixture"})
        folder = root / "bundle"
        save_bundle(bundle, baseline, folder)
        mlflow.log_artifacts(str(folder), artifact_path="model")
        run_id = run.info.run_id
    register_champion(tracking, run_id, name=MODEL_NAME)
    return run_id
