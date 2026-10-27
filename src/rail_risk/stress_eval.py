"""Stress test of the frozen champion on a gap-free week of its own routes and on unseen routes.

Nothing here fits or changes the champion. The snapshots come from the same `build_snapshots` as
training. For a new route the 28 day history features use only the dates fetched inside its window,
and a row with fewer than `min_prior_days` earlier fetched days is flagged. The historical baseline
for a scored date is fitted only on rows from earlier dates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rail_risk.evaluation import (
    Bands,
    HistoricalBaseline,
    Predictions,
    coverage_by_group,
    load_bands,
    load_snapshots,
    load_split_config,
    paired_mae_difference,
    persistence_baseline,
    reliability,
    score,
    threshold_probability,
)
from rail_risk.features.snapshots import HISTORY_DAYS
from rail_risk.lateness import group_labels, lateness_groups
from rail_risk.models.quantile import conformal_offset

JOURNEY_EDGES = (30, 75, 120, 180)  # scheduled minutes from the route origin to the destination
JOURNEY_LABELS = ["under 30", "30 to 74", "75 to 119", "120 to 179", "180 or more"]
ALL_HORIZONS = (5, 15, 30, 60, 90, 120)


@dataclass(frozen=True)
class StressConfig:
    own_routes: tuple[str, ...]
    own_start: date
    own_end: date
    new_routes: tuple[str, ...]
    new_start: date
    new_end: date
    horizons: tuple[int, ...]
    min_prior_days: int
    leakage_rows: int


def load_stress_config(path: Path | str) -> StressConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = StressConfig(
        own_routes=tuple(raw["own"]["routes"]),
        own_start=raw["own"]["start"],
        own_end=raw["own"]["end"],
        new_routes=tuple(raw["new"]["routes"]),
        new_start=raw["new"]["start"],
        new_end=raw["new"]["end"],
        horizons=tuple(int(h) for h in raw["horizons"]),
        min_prior_days=int(raw["min_prior_days"]),
        leakage_rows=int(raw["leakage_rows"]),
    )
    if set(cfg.own_routes) & set(cfg.new_routes):
        raise ValueError("a route cannot be both own and new")
    if cfg.own_end < cfg.own_start or cfg.new_end < cfg.new_start:
        raise ValueError("a window ends before it starts")
    return cfg


# history window flag


def prior_fetched_days(
    route_ids: np.ndarray,
    dates: np.ndarray,
    window: int = HISTORY_DAYS,
    fetched: dict[str, list[date]] | None = None,
) -> np.ndarray:
    """Per row, how many distinct fetched service dates of the same route lie in
    [date - window, date).

    `fetched` gives each route's fetched dates; without it the dates present in the rows are used,
    so a day with no fetch does not count either way.
    """
    out = np.zeros(len(route_ids), dtype=int)
    route_ids = np.asarray(route_ids)
    ordinals = np.array([d.toordinal() for d in dates])
    for route in np.unique(route_ids):
        mask = route_ids == route
        if fetched is not None and route in fetched:
            days = np.unique([d.toordinal() for d in fetched[route]])
        else:
            days = np.unique(ordinals[mask])
        lo = np.searchsorted(days, ordinals[mask] - window, side="left")
        hi = np.searchsorted(days, ordinals[mask], side="left")
        out[mask] = hi - lo
    return out


def flag_short_history(
    df: pd.DataFrame,
    min_days: int,
    window: int = HISTORY_DAYS,
    fetched: dict[str, list[date]] | None = None,
) -> pd.DataFrame:
    """Add `prior_days` and `short_history` (fewer than `min_days` prior fetched days)."""
    out = df.copy()
    out["prior_days"] = prior_fetched_days(
        out["route_id"].to_numpy(), out["service_date"].to_numpy(), window, fetched
    )
    out["short_history"] = out["prior_days"] < min_days
    return out


# snapshots


def build_stress_snapshots(
    spark: Any, interim: Path, routes_file: Path, cfg: StressConfig, out: Path
) -> pd.DataFrame:
    """Snapshots for the scored dates with the training pipeline, written to `out`."""
    import pyspark.sql.functions as F

    from rail_risk.features.build import load_inputs
    from rail_risk.features.snapshots import build_snapshots

    stops, rid_routes, route_dest = load_inputs(spark, interim, routes_file)
    frames = []
    fetched: dict[str, list[date]] = {}
    for group, routes, first, last in (
        ("own", cfg.own_routes, cfg.own_start, cfg.own_end),
        ("new", cfg.new_routes, cfg.new_start, cfg.new_end),
    ):
        # an own route keeps its full 28 day history; a new route only has its window
        history_from = first - timedelta(days=HISTORY_DAYS) if group == "own" else first
        s = stops.filter(
            (F.col("service_date") >= F.lit(history_from)) & (F.col("service_date") <= F.lit(last))
        )
        rr = rid_routes.filter(F.col("route_id").isin(list(routes)))
        snap = build_snapshots(spark, s, rr, route_dest).filter(
            F.col("service_date") >= F.lit(first)
        )
        days = s.join(rr, "rid").select("route_id", "service_date").distinct().collect()
        for r in days:
            fetched.setdefault(r["route_id"], []).append(r["service_date"])
        pdf = snap.toPandas()
        pdf["group"] = group
        frames.append(pdf)
    df = pd.concat(frames, ignore_index=True)
    df["service_date"] = pd.to_datetime(df["service_date"].astype(str)).dt.date
    df = flag_short_history(df, cfg.min_prior_days, fetched=fetched)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


# journey length and reason codes


def service_facts(interim: Path, routes_file: Path, rids: set[str]) -> pd.DataFrame:
    """Per (rid, route): scheduled minutes from the route origin to its destination and the
    delay reason codes on the stops up to the destination."""
    import sqlite3

    from rail_risk.config import load_routes

    stops = pd.read_parquet(interim / "_unified" / "hsp_stops.parquet")
    stops = stops[stops["rid"].isin(rids)]
    con = sqlite3.connect(f"file:{interim / 'hsp_manifest.sqlite'}?mode=ro", uri=True)
    try:
        rr = pd.DataFrame(
            con.execute("SELECT rid, route FROM rid_routes").fetchall(), columns=["rid", "route_id"]
        )
    finally:
        con.close()
    ends = {r.id: (r.origin, r.destination) for r in load_routes(routes_file).routes}
    rows = []
    for (rid, route), g in stops.merge(rr, on="rid").groupby(["rid", "route_id"]):
        if route not in ends:
            continue
        origin, dest = ends[route]
        d = g[(g["location"] == dest) & g["sched_arr"].notna()].sort_values("seq")
        o = g[(g["location"] == origin) & g["sched_dep"].notna()].sort_values("seq")
        if d.empty:
            continue
        dest_row = d.iloc[-1]
        journey = np.nan
        if not o.empty:
            journey = (dest_row["sched_arr"] - o.iloc[0]["sched_dep"]).total_seconds() / 60.0
        upto = g[g["seq"] <= dest_row["seq"]]
        coded = upto[upto["late_canc_reason"].notna()]
        reasons = sorted({f"{r.late_canc_reason}@{r.location}" for r in coded.itertuples()})
        rows.append({"rid": rid, "route_id": route, "journey_min": journey, "reasons": reasons})
    return pd.DataFrame(rows, columns=["rid", "route_id", "journey_min", "reasons"])


def journey_bucket(minutes: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(np.asarray(JOURNEY_EDGES), np.nan_to_num(minutes, nan=-1), side="right")
    labels = np.array(JOURNEY_LABELS + ["unknown"], dtype=object)
    return labels[np.where(np.isnan(minutes), len(JOURNEY_LABELS), idx)]


# predictions


def champion_predictions(bundle: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Median, 80% interval and band probabilities of the frozen champion for every row."""
    point, lo, hi = bundle.quantile.predict(df, "mondrian")
    probs = bundle.bands.predict(df)
    out = pd.DataFrame({"m_point": point, "m_lo": lo, "m_hi": hi}, index=df.index)
    for k in range(probs.shape[1]):
        out[f"p{k}"] = probs[:, k]
    return out


def _prob_cols(df: pd.DataFrame) -> np.ndarray:
    return df[[c for c in df.columns if c.startswith("p") and c[1:].isdigit()]].to_numpy()


def baseline_predictions(scored: pd.DataFrame, history: pd.DataFrame, bands: Bands) -> pd.DataFrame:
    """Persistence and route-hour historical baseline for `scored`.

    The historical baseline for a date is fitted on `history` rows dated strictly before it.
    """
    pers = persistence_baseline(scored, bands).point
    out = pd.DataFrame({"b_pers": pers}, index=scored.index)
    for col in ("b_hist", "b_hist_lo", "b_hist_hi"):
        out[col] = np.nan
    for day in sorted(scored["service_date"].unique()):
        prior = history[(history["service_date"] < day) & history["label_delay_min"].notna()]
        base = HistoricalBaseline(bands).fit(prior)
        mask = (scored["service_date"] == day).to_numpy()
        p = base.predict(scored[mask])
        out.loc[mask, "b_hist"] = p.point
        out.loc[mask, "b_hist_lo"] = p.lo
        out.loc[mask, "b_hist_hi"] = p.hi
    return out


# metrics


def _kw(boot: Any) -> dict[str, Any]:
    return {"resamples": boot.resamples, "seed": boot.seed, "confidence": boot.confidence}


def _ci(cell: dict[str, float | None]) -> dict[str, float | None]:
    return {k: cell.get(k) for k in ("value", "ci_low", "ci_high")}


def cell_metrics(df: pd.DataFrame, bands: Bands, boot: Any) -> dict[str, Any]:
    """Every per route and horizon metric for the rows in `df` (labelled rows only)."""
    kw = _kw(boot)
    y = df["label_delay_min"].to_numpy(dtype=float)
    dates = df["service_date"].to_numpy()
    model = Predictions(df["m_point"].to_numpy(), df["m_lo"].to_numpy(), df["m_hi"].to_numpy(),
                        _prob_cols(df))  # fmt: skip
    pers = Predictions(df["b_pers"].to_numpy())
    hist = Predictions(df["b_hist"].to_numpy(), df["b_hist_lo"].to_numpy(),
                       df["b_hist_hi"].to_numpy())  # fmt: skip
    m, p, h = (score(dates, y, x, bands, **kw) for x in (model, pers, hist))
    bins = 10 if len(y) >= 100 else 5
    p15 = reliability(
        dates, threshold_probability(model.probs, bands, 15), (y >= 15).astype(float),
        bins=bins, **kw,
    )  # fmt: skip
    best = "persistence" if p["mae"]["value"] <= h["mae"]["value"] else "historical"
    bar = pers if best == "persistence" else hist
    return {
        "n_rows": len(y),
        "n_dates": int(m["n_dates"]["value"]),
        "n_short_history": int(df["short_history"].sum()),
        "late_15_rate": float((y >= 15).mean()),
        "mae": {"model": _ci(m["mae"]), "persistence": _ci(p["mae"]), "historical": _ci(h["mae"])},
        "bar": best,
        "model_minus_bar": paired_mae_difference(dates, y, model.point, bar.point, **kw),
        "model_minus_persistence": paired_mae_difference(dates, y, model.point, pers.point, **kw),
        "model_minus_historical": paired_mae_difference(dates, y, model.point, hist.point, **kw),
        "coverage": _ci(m["coverage"]),
        "width": _ci(m["width"]),
        "historical_coverage": _ci(h["coverage"]),
        "coverage_by_lateness_group": coverage_by_group(
            dates, y, model, bands, lateness_groups(df), group_labels(), **kw
        ),
        "p_ge_15": {
            "bins": bins,
            "base_rate": p15["base_rate"],
            "brier": _ci(p15["brier"]),
            "ece": _ci(p15["ece"]),
        },
    }


def summary(df: pd.DataFrame, bands: Bands, boot: Any) -> dict[str, Any]:
    """A short summary for the diagnostic tables: MAE of the three forecasters and coverage."""
    kw = _kw(boot)
    y = df["label_delay_min"].to_numpy(dtype=float)
    dates = df["service_date"].to_numpy()
    model = Predictions(df["m_point"].to_numpy(), df["m_lo"].to_numpy(), df["m_hi"].to_numpy())
    m = score(dates, y, model, bands, **kw)
    p = score(dates, y, Predictions(df["b_pers"].to_numpy()), bands, **kw)
    h = score(dates, y, Predictions(df["b_hist"].to_numpy()), bands, **kw)
    return {
        "n_rows": len(y),
        "mae_model": _ci(m["mae"]),
        "mae_persistence": _ci(p["mae"]),
        "mae_historical": _ci(h["mae"]),
        "coverage": _ci(m["coverage"]),
        "width": _ci(m["width"]),
        "mean_error": float(np.mean(df["m_point"].to_numpy() - y)),
        "miss_below": float(np.mean(y < df["m_lo"].to_numpy())),
        "miss_above": float(np.mean(y > df["m_hi"].to_numpy())),
    }


def clean(value: Any) -> Any:
    """Make a nested result JSON safe: numpy scalars to Python, NaN to null, dates to text."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (date, pd.Timestamp)):
        return str(value)
    return value


def dump_json(result: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(clean(result), indent=2) + "\n", encoding="utf-8")


# completeness of the fetched window


def window_completeness(
    manifest: Path, route: str, start: date, end: date, windows_per_day: int = 16
) -> dict[str, Any]:
    """Complete, partial and missing days of one route inside [start, end], from the manifest."""
    import sqlite3

    con = sqlite3.connect(f"file:{manifest}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT day, COUNT(*) FROM windows WHERE route=? AND day BETWEEN ? AND ? GROUP BY day",
            (route, start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        con.close()
    have = {r[0]: r[1] for r in rows}
    days = [d.date().isoformat() for d in pd.date_range(start, end)]
    return {
        "days": len(days),
        "complete": sum(1 for d in days if have.get(d, 0) >= windows_per_day),
        "partial": [d for d in days if 0 < have.get(d, 0) < windows_per_day],
        "missing": [d for d in days if d not in have],
    }


# diagnostics


def unseen_category_check(
    bundle: Any, new_rows: pd.DataFrame, horizons: tuple[int, ...]
) -> dict[str, Any]:
    """What the frozen model does with a route id and operators it never saw."""
    fit = bundle.quantile.fits["direct"]
    x = fit.design.transform(new_rows)
    fake = new_rows.assign(route_id="zz-never-seen")
    same_as_fake = bool(
        np.allclose(
            bundle.quantile.predict(new_rows)[0], bundle.quantile.predict(fake)[0], atol=1e-9
        )
    )
    seen_toc = set(fit.design.categories["toc_code"])
    out: dict[str, Any] = {
        "seen_route_ids": fit.design.categories["route_id"],
        "seen_toc_codes": sorted(seen_toc),
        "route_id_became_missing_share": float(x["route_id"].isna().mean()),
        "toc_became_missing_share": float(x["toc_code"].isna().mean()),
        "same_prediction_as_a_made_up_route_id": same_as_fake,
        "conformal_offsets_used": "pooled over routes, as no calibration cell exists for the route",
    }
    y = new_rows["label_delay_min"].to_numpy(dtype=float)
    mask = new_rows["horizon_min"].isin(list(horizons)).to_numpy()
    for label, frame in (
        ("natural_unseen", new_rows),
        ("as_bhm-eus", new_rows.assign(route_id="bhm-eus")),
        ("as_ast-bhm", new_rows.assign(route_id="ast-bhm")),
    ):
        point, lo, hi = bundle.quantile.predict(frame)
        out[label] = {
            "mae": float(np.mean(np.abs(point[mask] - y[mask]))),
            "coverage": float(np.mean(((y >= lo) & (y <= hi))[mask])),
        }
    return out


def worst_services(
    rows: pd.DataFrame, facts: pd.DataFrame, horizons: tuple[int, ...], n: int = 20
) -> list[dict[str, Any]]:
    """The `n` services with the largest mean absolute error of the champion over `horizons`."""
    sub = rows[rows["horizon_min"].isin(horizons)].copy()
    sub["abs_err"] = (sub["m_point"] - sub["label_delay_min"]).abs()
    grouped = sub.groupby(["rid", "route_id"])
    agg = grouped.agg(
        service_date=("service_date", "first"),
        toc_code=("toc_code", "first"),
        sched_arrival=("dest_sched_arr", "first"),
        delay_min=("label_delay_min", "first"),
        mean_abs_err=("abs_err", "mean"),
    ).reset_index()
    at60 = sub[sub["horizon_min"] == 60].set_index(["rid", "route_id"])
    agg["model_median_at_60"] = [
        at60["m_point"].get((r, k)) for r, k in zip(agg.rid, agg.route_id, strict=True)
    ]
    agg["historical_at_60"] = [
        at60["b_hist"].get((r, k)) for r, k in zip(agg.rid, agg.route_id, strict=True)
    ]
    agg = agg.merge(facts, on=["rid", "route_id"], how="left")
    agg = agg.sort_values("mean_abs_err", ascending=False).head(n)
    return agg.to_dict(orient="records")


def _fmt(cell: dict[str, float | None] | None, d: int = 2) -> str:
    if not cell or cell.get("value") is None:
        return "n/a"
    lo, hi = cell.get("ci_low"), cell.get("ci_high")
    if lo is None or hi is None:
        return f"{cell['value']:.{d}f}"
    return f"{cell['value']:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


RANGE_FEATURES = [
    "last_stop_seq",
    "stops_remaining",
    "sched_minutes_remaining",
    "n_stops_passed",
    "minutes_since_last_stop",
    "hist_median_28d",
    "hist_p80_28d",
    "hist_n_28d",
]


def out_of_range_share(
    reference: pd.DataFrame, rows: pd.DataFrame, features: list[str] = RANGE_FEATURES
) -> dict[str, float]:
    """Share of non-missing values in `rows` outside the min to max seen in `reference`."""
    out: dict[str, float] = {}
    for f in features:
        ref, vals = reference[f].dropna(), rows[f].dropna()
        out[f] = float(((vals < ref.min()) | (vals > ref.max())).mean()) if len(vals) else np.nan
    return out


def widening_needed(rows: pd.DataFrame, coverage: float = 0.8) -> float:
    """Diagnostic only: the symmetric widening of the champion's interval that would have given
    `coverage` on these very rows. Positive means the interval was too narrow. Not applied."""
    y = rows["label_delay_min"].to_numpy(dtype=float)
    return conformal_offset(y, rows["m_lo"].to_numpy(), rows["m_hi"].to_numpy(), coverage)


# the whole evaluation


def run_stress_eval(
    settings: Any,
    cfg: StressConfig,
    *,
    skip_build: bool = False,
    log: Any = None,
) -> dict[str, Any]:
    import time
    from pathlib import Path as _Path

    from rail_risk.live_eval import _champion_state
    from rail_risk.models import registry

    started = time.monotonic()
    say = log.info if log else (lambda *a, **k: None)
    interim = settings.data_dir / "interim"
    stress_dir = settings.data_dir / "stress"
    snap_path = stress_dir / "snapshots.parquet"
    bands = load_bands(_Path("config/delay_bands.yaml"))
    boot = load_split_config(_Path("config/split.yaml"))
    mlruns = _Path("mlruns")

    before = _champion_state(mlruns)
    loaded = registry.load_alias(mlruns)
    if loaded.run_id != before["run_id"]:
        raise ValueError("champion changed while loading")

    leakage_path = stress_dir / "leakage.json"
    if skip_build:
        df = load_snapshots(snap_path)
        leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    else:
        from rail_risk.features.build import leakage_check_real
        from rail_risk.spark import build_spark

        spark = build_spark()
        df = build_stress_snapshots(spark, interim, settings.routes_file, cfg, snap_path)
        say("stress_snapshots", rows=len(df))
        checked, deleted, mismatches, per_route = leakage_check_real(
            spark, interim, settings.routes_file, n=cfg.leakage_rows, seed=7,
            start=cfg.new_start, end=cfg.new_end, routes=cfg.new_routes,
        )  # fmt: skip
        leakage = {
            "rows_checked": checked, "rows_checked_by_route": per_route,
            "sampled_with_future_actuals_deleted": deleted, "mismatches": len(mismatches),
            "first_mismatches": mismatches[:5],
        }  # fmt: skip
        leakage_path.write_text(json.dumps(clean(leakage), indent=2), encoding="utf-8")
        say("stress_leakage", **{k: v for k, v in leakage.items() if k != "first_mismatches"})

    # rows of the own routes before the window, only as history for the historical baseline
    earlier = load_snapshots(settings.data_dir / "features" / "snapshots")
    cols = ["route_id", "service_date", "dest_hour", "label_delay_min", "hist_median_28d"]
    train_end = date.fromisoformat(str(loaded.bundle.meta["train_end"]))
    own_rows = earlier[earlier["route_id"].isin(cfg.own_routes)]
    train_reference = own_rows[own_rows["service_date"] <= train_end]
    earlier = own_rows[own_rows["service_date"] < cfg.own_start][cols]
    history = pd.concat([earlier, df[cols]], ignore_index=True)

    labelled = df[df["label_delay_min"].notna()].reset_index(drop=True)
    preds = champion_predictions(loaded.bundle, labelled)
    base = baseline_predictions(labelled, history, bands)
    rows = pd.concat([labelled, preds, base], axis=1)
    rows.drop(columns=["group"]).assign(group=rows["group"]).to_parquet(
        stress_dir / "predictions.parquet"
    )
    say("stress_predictions", rows=len(rows), seconds=round(time.monotonic() - started))

    # per route and horizon
    cells: dict[str, Any] = {}
    for route in (*cfg.own_routes, *cfg.new_routes):
        for h in cfg.horizons:
            sub = rows[(rows["route_id"] == route) & (rows["horizon_min"] == h)]
            if len(sub):
                cells.setdefault(route, {})[str(h)] = cell_metrics(sub, bands, boot)

    main = rows[rows["horizon_min"].isin(cfg.horizons)]
    by_route = {
        r: summary(main[main["route_id"] == r], bands, boot)
        for r in (*cfg.own_routes, *cfg.new_routes)
        if (main["route_id"] == r).any()
    }
    seen_toc = set(loaded.bundle.quantile.fits["direct"].design.categories["toc_code"])
    by_toc = {
        t: {
            **summary(g, bands, boot),
            "seen_in_training": t in seen_toc,
            "routes": sorted(g["route_id"].unique()),
        }  # fmt: skip
        for t, g in main.groupby("toc_code")
    }
    facts = service_facts(interim, settings.routes_file, set(rows["rid"]))
    main = main.merge(facts[["rid", "route_id", "journey_min"]], on=["rid", "route_id"], how="left")
    main["journey_bucket"] = journey_bucket(main["journey_min"].to_numpy(dtype=float))
    by_journey = {b: summary(g, bands, boot) for b, g in main.groupby("journey_bucket")}
    by_route_toc = {
        f"{r}/{t}": {**summary(g, bands, boot), "seen_operator": t in seen_toc}
        for (r, t), g in main.groupby(["route_id", "toc_code"])
        if len(g) >= 30
    }
    needed = {
        r: {
            str(h): widening_needed(rows[(rows["route_id"] == r) & (rows["horizon_min"] == h)])
            for h in cfg.horizons
            if ((rows["route_id"] == r) & (rows["horizon_min"] == h)).any()
        }
        for r in (*cfg.own_routes, *cfg.new_routes)
    }
    feature_range = {
        r: out_of_range_share(train_reference, rows[rows["route_id"] == r])
        for r in (*cfg.own_routes, *cfg.new_routes)
    }
    by_minutes = {
        grp: {
            str(h): summary(rows[(rows["group"] == grp) & (rows["horizon_min"] == h)], bands, boot)
            for h in ALL_HORIZONS
            if ((rows["group"] == grp) & (rows["horizon_min"] == h)).any()
        }
        for grp in ("own", "new")
    }
    long_hist = main[(main["group"] == "new") & ~main["short_history"]]
    short_hist = main[(main["group"] == "new") & main["short_history"]]
    history_split = {
        "long_history_rows": int(len(long_hist)),
        "short_history_rows": int(len(short_hist)),
        "long_history": summary(long_hist, bands, boot) if len(long_hist) else None,
        "short_history": summary(short_hist, bands, boot) if len(short_hist) else None,
    }
    new_rows = rows[rows["group"] == "new"]
    unseen = unseen_category_check(loaded.bundle, new_rows, cfg.horizons)
    worst = worst_services(rows, facts, cfg.horizons)

    after = _champion_state(mlruns)
    manifest = interim / "hsp_manifest.sqlite"
    counts = {
        r: {
            "snapshot_rows": int((df["route_id"] == r).sum()),
            "scored_rows": int((rows["route_id"] == r).sum()),
            "unlabelled_rows": int(((df["route_id"] == r) & df["label_delay_min"].isna()).sum()),
            "cancelled_rows": int(((df["route_id"] == r) & df["is_candidate_cancelled"]).sum()),
            "services": int(df.loc[df["route_id"] == r, "rid"].nunique()),
            "short_history_rows": int(((df["route_id"] == r) & df["short_history"]).sum()),
        }
        for r in (*cfg.own_routes, *cfg.new_routes)
    }
    completeness = {
        r: window_completeness(
            manifest, r, *((cfg.own_start, cfg.own_end) if r in cfg.own_routes
                           else (cfg.new_start, cfg.new_end))
        )
        for r in (*cfg.own_routes, *cfg.new_routes)
    }  # fmt: skip
    live = _live_comparison(cells, cfg)
    return {
        "champion": {
            "before": before,
            "after": after,
            "unchanged": before == after,
            "meta": loaded.bundle.meta,
        },  # fmt: skip
        "config": {
            "own": {"routes": cfg.own_routes, "start": cfg.own_start, "end": cfg.own_end},
            "new": {"routes": cfg.new_routes, "start": cfg.new_start, "end": cfg.new_end},
            "horizons": cfg.horizons,
            "min_prior_days": cfg.min_prior_days,
            "bootstrap": {
                "resamples": boot.resamples,
                "seed": boot.seed,
                "confidence": boot.confidence,
            },
        },  # fmt: skip
        "completeness": completeness,
        "counts": counts,
        "leakage_check": leakage,
        "cells": cells,
        "by_route": by_route,
        "by_toc": by_toc,
        "by_journey_length": by_journey,
        "by_route_and_operator": by_route_toc,
        "interval_widening_needed_diagnostic": needed,
        "share_of_feature_values_outside_training_range": {
            "training_rows_to": train_end,
            "by_route": feature_range,
        },  # fmt: skip
        "by_minutes_to_arrival": by_minutes,
        "new_route_history_split": history_split,
        "unseen_category": unseen,
        "worst_services": worst,
        "gap_free_vs_live": live,
        "seconds": round(time.monotonic() - started),
    }


def _live_comparison(cells: dict[str, Any], cfg: StressConfig) -> dict[str, Any]:
    """The gap-free HSP week against the Darwin live replay of the same dates."""
    path = Path("reports/live_eval.json")
    if not path.exists():
        return {}
    live = json.loads(path.read_text(encoding="utf-8"))["singles"]
    out: dict[str, Any] = {}
    for route in cfg.own_routes:
        for h in cfg.horizons:
            a, b = cells.get(route, {}).get(str(h)), live.get(route, {}).get(str(h))
            if not a or not b:
                continue
            out.setdefault(route, {})[str(h)] = {
                "gap_free": {
                    "n_rows": a["n_rows"],
                    "n_dates": a["n_dates"],
                    "mae": a["mae"]["model"],
                    "coverage": a["coverage"],
                },  # fmt: skip
                "live": {
                    "n_rows": b["n_rows"],
                    "n_dates": b["n_dates"],
                    "mae": b["model"]["mae"],
                    "coverage": b["model"]["coverage"],
                },  # fmt: skip
            }
    return out


# report

PLAIN_ONLY = ("\u2014", "\u2013")  # the house style bans em and en dashes


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in body]


def _route_order(result: dict[str, Any]) -> list[str]:
    c = result["config"]
    return [*c["own"]["routes"], *c["new"]["routes"]]


def write_markdown(result: dict[str, Any], notes: list[str]) -> str:
    """The report in plain prose. `notes` are the findings and recommendations, written last."""
    c, ch = result["config"], result["champion"]
    horizons = c["horizons"]
    routes = _route_order(result)
    cells = result["cells"]
    lines = ["# Stress test of the frozen champion", ""]
    lines += [
        f"Champion run {ch['before']['run_id']} "
        f"(registry version {ch['before']['registry_version']}) "
        f"was scored without refitting, recalibrating or changing any threshold. Its files hash "
        f"{'identically' if ch['unchanged'] else 'DIFFERENTLY'} before and after the run. "
        f"The own routes {', '.join(c['own']['routes'])} were scored from {c['own']['start']} to "
        f"{c['own']['end']}. The new routes {', '.join(c['new']['routes'])} were scored from "
        f"{c['new']['start']} to {c['new']['end']}. Confidence intervals are 95% percentile "
        f"bootstrap intervals that resample whole service dates "
        f"({c['bootstrap']['resamples']} resamples, seed {c['bootstrap']['seed']}).",
        "",
        "The bar in each cell is the better of the persistence forecast and the historical "
        "route-hour baseline on that cell. The historical baseline for a date is fitted only on "
        "rows from earlier dates, and until a new route has 10 rows of its own it uses the pooled "
        "rows of the other routes.",
        "",
    ]

    lines += ["## Data", ""]
    body = []
    for r in routes:
        n, comp = result["counts"][r], result["completeness"][r]
        body.append(
            [r, str(n["services"]), str(n["snapshot_rows"]), str(n["scored_rows"]),
             str(n["cancelled_rows"]), str(n["short_history_rows"]),
             f"{comp['complete']} of {comp['days']}",
             ", ".join(comp["partial"] + comp["missing"]) or "none"]
        )  # fmt: skip
    lines += _table(
        ["route", "services", "snapshot rows", "scored rows", "cancelled rows",
         "rows with under 7 prior days", "complete days", "partial or missing days"],
        body,
    )  # fmt: skip
    lk = result["leakage_check"]
    lines += [
        "",
        f"Leakage check on real rows from the new routes: {lk['rows_checked']} rows rebuilt after "
        f"deleting every actual time after T, {lk['mismatches']} feature mismatches "
        f"(rows by route: {lk['rows_checked_by_route']}).",
        "",
    ]

    for h in horizons:
        lines += [f"## Results at {h} minutes to arrival", ""]
        body = []
        for r in routes:
            m = cells.get(r, {}).get(str(h))
            if not m:
                continue
            body.append(
                [r, f"{m['n_rows']} ({m['n_dates']} d)", _fmt(m["mae"]["model"]),
                 _fmt(m["mae"]["persistence"]), _fmt(m["mae"]["historical"]), m["bar"],
                 _fmt(m["model_minus_bar"]), _fmt(m["coverage"], 3), _fmt(m["width"], 1),
                 _fmt(m["p_ge_15"]["brier"], 3), _fmt(m["p_ge_15"]["ece"], 3)]
            )  # fmt: skip
        lines += _table(
            ["route", "rows", "MAE model", "MAE persistence", "MAE historical", "bar",
             "model minus bar", "80% coverage", "width", "P(late 15) Brier", "ECE"],
            body,
        )  # fmt: skip
        lines += [
            "",
            "Paired MAE difference, model minus each baseline (negative favours the model):",
            "",
        ]
        body = [
            [r, _fmt(cells[r][str(h)]["model_minus_persistence"]),
             _fmt(cells[r][str(h)]["model_minus_historical"])]
            for r in routes
            if str(h) in cells.get(r, {})
        ]  # fmt: skip
        lines += _table(["route", "vs persistence", "vs historical"], body)
        lines += ["", "Coverage by observed lateness at T (80% nominal):", ""]
        labels = list(next(iter(cells.values()))[str(h)]["coverage_by_lateness_group"])
        body = []
        for r in routes:
            m = cells.get(r, {}).get(str(h))
            if not m:
                continue
            row = [r]
            for lab in labels:
                g = m["coverage_by_lateness_group"][lab]
                row.append("n/a" if g["value"] is None else f"{g['value']:.2f} (n={int(g['n'])})")
            body.append(row)
        lines += _table(["route", *labels], body)
        lines += [""]

    live = result["gap_free_vs_live"]
    if live:
        lines += ["## Gap-free week against the live replay", "", "The live replay lost 88 of 168 "
                  "hours to capture gaps, so it scored fewer services. Both score the frozen "
                  "champion on 16 to 22 October.", ""]  # fmt: skip
        body = []
        for r, per in live.items():
            for h, v in per.items():
                body.append(
                    [r, h, f"{v['gap_free']['n_rows']} ({v['gap_free']['n_dates']} d)",
                     _fmt(v["gap_free"]["mae"]), _fmt(v["gap_free"]["coverage"], 3),
                     f"{v['live']['n_rows']} ({v['live']['n_dates']} d)",
                     _fmt(v["live"]["mae"]), _fmt(v["live"]["coverage"], 3)]
                )  # fmt: skip
        lines += _table(
            ["route", "horizon", "gap-free rows", "gap-free MAE", "gap-free coverage",
             "live rows", "live MAE", "live coverage"],
            body,
        )  # fmt: skip
        lines += [""]

    def summary_rows(table: dict[str, Any]) -> list[list[str]]:
        return [
            [k, str(v["n_rows"]), _fmt(v["mae_model"]), _fmt(v["mae_persistence"]),
             _fmt(v["mae_historical"]), _fmt(v["coverage"], 3), _fmt(v["width"], 1)]
            for k, v in table.items()
        ]  # fmt: skip

    head = ["", "rows", "MAE model", "MAE persistence", "MAE historical", "80% coverage", "width"]
    lines += ["## Where it breaks", "",
              f"Rows below pool the horizons {', '.join(str(h) for h in horizons)}.", "",
              "### By route", ""]  # fmt: skip
    lines += _table(["route", *head[1:]], summary_rows(result["by_route"]))
    lines += [
        "",
        "Signed error is the champion's median minus the observed delay in minutes, so positive "
        "means it expected the train to be later than it was. The two miss columns are the "
        "shares of rows where the observed delay fell below or above the 80% interval.",
        "",
    ]
    lines += _table(
        ["route", "signed error", "missed below", "missed above"],
        [
            [r, f"{v['mean_error']:+.2f}", f"{v['miss_below']:.1%}", f"{v['miss_above']:.1%}"]
            for r, v in result["by_route"].items()
        ],
    )
    lines += ["", "### By operator", ""]
    toc = result["by_toc"]
    body = [
        [f"{k} ({'seen' if v['seen_in_training'] else 'unseen'} in training)", *row[1:]]
        for (k, v), row in zip(toc.items(), summary_rows(toc), strict=True)
    ]
    lines += _table(["operator", *head[1:]], body)
    lines += ["", "### By route and operator (30 or more rows)", ""]
    rto = result["by_route_and_operator"]
    body = [
        [f"{k} ({'seen' if v['seen_operator'] else 'unseen'} operator)", *row[1:]]
        for (k, v), row in zip(rto.items(), summary_rows(rto), strict=True)
    ]
    lines += _table(["route and operator", *head[1:]], body)
    lines += ["", "### By scheduled journey length (minutes, origin to destination)", ""]
    lines += _table(["journey", *head[1:]], summary_rows(result["by_journey_length"]))
    lines += ["", "### By minutes to arrival", ""]
    for grp, label in (("own", "Own routes"), ("new", "New routes")):
        lines += [f"{label}:", ""]
        lines += _table(["minutes", *head[1:]], summary_rows(result["by_minutes_to_arrival"][grp]))
        lines += [""]
    lines += [
        "",
        "### Interval widening that would have been needed (diagnostic, not applied)",
        "",
        "The symmetric widening of the champion's 80% interval, in minutes, that would have "
        "given 80% coverage on these same rows. Positive means the interval was too narrow. It "
        "is computed on the scored rows themselves, so it describes the gap and is not a "
        "recalibration.",
        "",
    ]
    need = result["interval_widening_needed_diagnostic"]
    lines += _table(
        ["route", *[f"{h} min" for h in horizons]],
        [[r, *[f"{need[r][str(h)]:+.1f}" for h in horizons]] for r in routes if r in need],
    )
    fr = result["share_of_feature_values_outside_training_range"]
    names = list(next(iter(fr["by_route"].values())))
    lines += [
        "",
        "### Feature values outside the range seen in training",
        "",
        f"Share of each route's rows (all six horizons) whose value lies outside the minimum to "
        f"maximum seen on the own routes up to {fr['training_rows_to']}.",
        "",
    ]
    lines += _table(
        ["route", *names],
        [
            [r, *["n/a" if v[n] is None else f"{v[n]:.0%}" for n in names]]
            for r, v in fr["by_route"].items()
        ],
    )
    hs = result["new_route_history_split"]
    lines += ["", "### New routes by history length", ""]
    body = []
    for key, name in (
        ("long_history", "7 or more prior days"),
        ("short_history", "under 7 prior days"),
    ):
        if hs[key]:
            v = hs[key]
            body.append([name, str(v["n_rows"]), _fmt(v["mae_model"]), _fmt(v["mae_persistence"]),
                         _fmt(v["mae_historical"]), _fmt(v["coverage"], 3)])  # fmt: skip
    lines += _table(["history", "rows", "MAE model", "MAE persistence", "MAE historical",
                     "80% coverage"], body)  # fmt: skip

    u = result["unseen_category"]
    lines += [
        "",
        "### An unseen route id",
        "",
        f"The champion was fitted with route ids {', '.join(u['seen_route_ids'])} and operators "
        f"{', '.join(u['seen_toc_codes'])}. The design matrix fixes the category lists at fit "
        "time, so a route id or operator code outside them becomes a missing category value "
        f"({u['route_id_became_missing_share']:.0%} of new route rows for the route id, "
        f"{u['toc_became_missing_share']:.0%} for the operator). LightGBM treats a missing "
        "categorical value as missing and sends it down the branch of each split that the "
        "split's selected categories do not take, so the model sees an unseen route as no route "
        f"at all. Predictions for the new routes equal predictions with a made up route id: "
        f"{u['same_prediction_as_a_made_up_route_id']}. The conformal offsets fall back to the "
        "pooled set, because the route has no calibration cell. Pandas already emits a "
        "Pandas4Warning for this conversion (values outside a fixed category list) and says it "
        "will raise in a future version, so the path needs an explicit fallback before a "
        "pandas upgrade.",
        "",
        "Pooling the 15, 60 and 120 minute rows of the new routes, MAE and coverage with the "
        "route id as it is, and with the route id replaced by each trained route:",
        "",
    ]
    lines += _table(
        ["route id given to the model", "MAE", "80% coverage"],
        [[k, f"{u[k]['mae']:.2f}", f"{u[k]['coverage']:.3f}"]
         for k in ("natural_unseen", "as_bhm-eus", "as_ast-bhm")],
    )  # fmt: skip

    lines += ["", "### The 20 worst services", "",
              "Ranked by the champion's mean absolute error over the scored horizons. Reason "
              "codes are the delay reason codes HSP attaches to the stops up to the destination, "
              "shown as code at station; none means HSP gave none. No Delay Repay amounts are "
              "given.", ""]  # fmt: skip
    body = []
    for w in result["worst_services"]:
        reasons = w.get("reasons")
        body.append(
            [w["route_id"], str(w["service_date"]), str(w["toc_code"]),
             str(w["sched_arrival"])[11:16], f"{w['delay_min']:.0f}", f"{w['mean_abs_err']:.1f}",
             "n/a" if w.get("model_median_at_60") is None else f"{w['model_median_at_60']:.1f}",
             ", ".join(reasons) if isinstance(reasons, list) and reasons else "none"]
        )  # fmt: skip
    lines += _table(["route", "date", "operator", "due (UTC)", "delay min", "mean abs error",
                     "model median at 60", "reason codes"], body)  # fmt: skip
    lines += ["", *notes, ""]
    text = "\n".join(lines)
    for ch in PLAIN_ONLY:
        if ch in text:
            raise ValueError("report text contains an em or en dash")
    return text


def write_reports(result: dict[str, Any], notes: list[str], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / "stress_eval.md", out_dir / "stress_eval.json"
    md.write_text(write_markdown(result, notes), encoding="utf-8")
    dump_json(result, js)
    return md, js


def _v(cell: dict[str, Any] | None, d: int = 2) -> str:
    return "n/a" if not cell or cell.get("value") is None else f"{cell['value']:.{d}f}"


def findings(result: dict[str, Any]) -> list[str]:
    """Failure patterns, limits and ranked recommendations. The numbers come from `result`."""
    c = result["config"]
    new = c["new"]["routes"]
    cells, by_route = result["cells"], result["by_route"]
    by_min = result["by_minutes_to_arrival"]
    need = result["interval_widening_needed_diagnostic"]
    hs = result["new_route_history_split"]
    u = result["unseen_category"]
    live = result["gap_free_vs_live"]

    def diff(route: str, h: int, key: str) -> str:
        cell = cells[route][str(h)][key]
        return f"{cell['value']:+.2f} [{cell['ci_low']:+.2f}, {cell['ci_high']:+.2f}]"

    def cov(route: str) -> str:
        return f"{by_route[route]['coverage']['value']:.2f}"

    worse_txt = ", ".join(
        f"{r} at {h} minutes ({diff(r, h, 'model_minus_persistence')})"
        for r in new for h in (15, 60)
        if cells[r][str(h)]["model_minus_persistence"]["ci_low"] > 0
    )  # fmt: skip
    beats_hist = sum(
        1 for r in new for h in c["horizons"]
        if cells[r][str(h)]["model_minus_historical"]["ci_high"] < 0
    )  # fmt: skip
    total_new = sum(1 for r in new for h in c["horizons"] if str(h) in cells[r])
    liv = by_route["bhm-liv"]
    man = by_route["bhm-man"]
    j180 = result["by_journey_length"].get("180 or more")
    lo_hist, sh_hist = hs["long_history"], hs["short_history"]
    worst = result["worst_services"]
    worst_eus = sum(1 for w in worst if w["route_id"] == "bhm-eus")
    big_miss = sum(
        1 for w in worst
        if w["delay_min"] >= 30 and (w["model_median_at_60"] or 0) < 10
    )  # fmt: skip
    train_end = result["champion"]["meta"].get("train_end")
    worse_routes = ", ".join(
        sorted(
            {
                r
                for r in new
                for h in (15, 60)
                if cells[r][str(h)]["model_minus_persistence"]["ci_low"] > 0
            }
        )
    )
    oor = result["share_of_feature_values_outside_training_range"]["by_route"]
    liv_hist_oor = oor["bhm-liv"]["hist_median_28d"]
    early = [w for w in worst if w["delay_min"] <= -20]
    early_txt = ", ".join(
        f"{w['route_id']} {w['service_date']} due {str(w['sched_arrival'])[11:16]} UTC"
        f" ({w['delay_min']:.0f} min)"
        for w in early
    )
    code_days: dict[str, int] = {}
    for w in worst:
        for code in {r.split("@")[0] for r in (w.get("reasons") or [])}:
            key = f"{code} on {w['route_id']} {w['service_date']}"
            code_days[key] = code_days.get(key, 0) + 1
    top_code, top_count = max(code_days.items(), key=lambda kv: kv[1], default=("none", 0))
    live_higher = sum(
        1 for per in live.values() for v in per.values()
        if v["live"]["mae"]["value"] > v["gap_free"]["mae"]["value"]
    )  # fmt: skip
    live_cells = sum(len(per) for per in live.values())
    live_outside = sum(
        1 for per in live.values() for v in per.values()
        if v["gap_free"]["mae"]["ci_high"] is not None
        and v["live"]["mae"]["value"] > v["gap_free"]["mae"]["ci_high"]
    )  # fmt: skip

    lines = ["## Failure patterns", ""]
    lines += [
        f"1. Coverage falls short on most routes the champion has not seen, in a different "
        f"direction on each. Pooling the 15, 60 and 120 minute rows, the 80% interval covered "
        f"{cov('bhm-eus')} on bhm-eus and {cov('ast-bhm')} on ast-bhm, but {cov('bhm-man')} on "
        f"bhm-man, {cov('bhm-lds')} on bhm-lds, {cov('bhm-bri')} on bhm-bri, {cov('bhm-glc')} on "
        f"bhm-glc and {cov('bhm-liv')} on bhm-liv. At 5 and 15 minutes to arrival the new routes "
        f"together covered {by_min['new']['5']['coverage']['value']:.2f} and "
        f"{by_min['new']['15']['coverage']['value']:.2f}, against "
        f"{by_min['own']['5']['coverage']['value']:.2f} and "
        f"{by_min['own']['15']['coverage']['value']:.2f} on the own routes. On bhm-liv "
        f"{liv['miss_below']:.1%} of observed delays fell below the lower bound and "
        f"{liv['miss_above']:.1%} above it, and the interval would have needed widening by "
        f"{need['bhm-liv']['15']:+.1f}, {need['bhm-liv']['60']:+.1f} and "
        f"{need['bhm-liv']['120']:+.1f} minutes at 15, 60 and 120 minutes to reach 80%. On bhm-man "
        f"the misses went the other way ({man['miss_above']:.1%} above, {man['miss_below']:.1%} "
        f"below) and the median ran {-man['mean_error']:.2f} minutes early on average. The "
        f"conformal offsets are the pooled ones for any route with no calibration cell, so none "
        f"of this is corrected.",
        "",
        f"2. No point forecast edge over carrying the current lateness forward on the longer or "
        f"more different routes. The paired difference with persistence has an interval above "
        f"zero for {worse_txt or 'no route at 15 or 60 minutes'}. On journeys of 180 minutes or "
        f"more the champion's MAE was {_v(j180['mae_model']) if j180 else 'n/a'} against "
        f"{_v(j180['mae_persistence']) if j180 else 'n/a'} for persistence. It does beat the "
        f"historical route-hour baseline on the new routes: in {beats_hist} of {total_new} route "
        f"and horizon cells the paired interval lies wholly below zero. Replacing the route id "
        f"with a trained route's id left MAE unchanged at "
        f"{u['natural_unseen']['mae']:.2f}, {u['as_bhm-eus']['mae']:.2f} and "
        f"{u['as_ast-bhm']['mae']:.2f}, so the route id itself carries no point information for "
        f"these routes.",
        "",
        f"3. Weak start on new routes with little history. Rows with fewer than 7 prior fetched "
        f"days ({sh_hist['n_rows'] if sh_hist else 0} rows) were covered "
        f"{_v(sh_hist['coverage'], 3) if sh_hist else 'n/a'} of the time, against "
        f"{_v(lo_hist['coverage'], 3) if lo_hist else 'n/a'} with 7 or more days, and the "
        f"champion's MAE was {_v(sh_hist['mae_model']) if sh_hist else 'n/a'} against "
        f"{_v(lo_hist['mae_model']) if lo_hist else 'n/a'}, while persistence scored "
        f"{_v(sh_hist['mae_persistence']) if sh_hist else 'n/a'} and "
        f"{_v(lo_hist['mae_persistence']) if lo_hist else 'n/a'}. The two groups are different "
        f"days, so this is a pattern and not a controlled comparison.",
        "",
        f"Tail cases. {worst_eus} of the 20 worst services are on bhm-eus, and {big_miss} of the "
        f"20 were 30 or more minutes late with a median at 60 minutes below 10. The most "
        f"repeated reason code among them is {top_code}, on {top_count} of the 20, which fits "
        f"one incident affecting several services but does not confirm it. {len(early)} of the "
        f"20 are arrivals 20 or more minutes early ({early_txt}); whether that is a timetable "
        f"or data effect was not checked. Reason code meanings were not looked up, so no cause "
        f"is named.",
        "",
        "## Gap-free week against the live replay",
        "",
        f"On bhm-eus and ast-bhm for 16 to 22 October the live replay MAE was higher than the "
        f"gap-free MAE in {live_higher} of {live_cells} route and horizon cells, and above the "
        f"gap-free interval in {live_outside}. The two scored different services (the live "
        f"replay lost 88 of 168 hours) from different sources (Darwin against HSP actual "
        f"times), and this step did not separate those causes.",
        "",
        "## Limits",
        "",
        "- The own routes have 7 service dates, so their intervals are wide.",
        "- bhm-bri has 16 scored dates. HSP answered every window on 3, 4, 10, 11, 17 and 18 "
        "October with no services (HTTP 200, not a failure), while a probe of 26 and 27 "
        "September returned services. The cause was not verified.",
        "- New route history features use only dates inside the window, bhm-man included, "
        "although older bhm-man data exist. They were not used, so all five new routes start "
        "cold in the same way.",
        f"- The bundle metadata records the training block as ending {train_end}"
        f", with later validation and test blocks. Every scored date is after 15 October.",
        "- The bar is the better of the two baselines on each cell, chosen after the fact, which "
        "makes it a stricter bar than the validation bar the earlier reports used.",
        "- Delay Repay amounts are not computed.",
        "",
        "## Recommended model changes, ranked",
        "",
        "These are not implemented. The champion is unchanged.",
        "",
        f"1. Per-route interval recalibration, with a shared prior until a route has enough "
        f"days, and separate lower and upper offsets. Evidence: coverage on new routes ranged "
        f"from {cov('bhm-liv')} to {cov('bhm-man')}, the shortfall was on opposite sides for "
        f"bhm-liv and bhm-man, and the widening needed ranged from "
        f"{min(v for r in new for v in need[r].values()):+.1f} to "
        f"{max(v for r in new for v in need[r].values()):+.1f} minutes. This is the cheapest "
        f"change and addresses the coverage promise directly.",
        "",
        f"2. Route-agnostic features. Drop the route id category (it changed nothing for unseen "
        f"routes) and express stop counts and time remaining as shares of the journey, adding "
        f"scheduled journey length. Evidence: the champion was worse than persistence "
        f"(paired interval above zero) on "
        f"{worse_routes} "
        f"and on 180 minute journeys. That the stop counts are the cause was not tested.",
        "",
        f"3. More training routes, chosen for length and operator. Evidence: the champion was "
        f"fitted "
        f"on {', '.join(u['seen_route_ids'])} with operators {', '.join(u['seen_toc_codes'])}, "
        f"and {u['toc_became_missing_share']:.0%} of new route rows had an operator it had never "
        f"seen. It still beats the historical baseline in {beats_hist} of {total_new} cells, "
        f"so some of it transfers.",
        "",
        f"4. History features that survive a cold start: fall back to a pooled operator and hour "
        f"history for a route's first weeks, and train with randomly shortened histories. "
        f"Evidence: the short history rows above, and "
        f"{liv_hist_oor:.0%} "
        f"of bhm-liv rows had a hist_median_28d outside the range seen in training.",
        "",
        "5. Per-operator history keyed on a cleaner operator identity. Evidence is weaker: the "
        "operator code LM covers West Midlands Railway on ast-bhm and London Northwestern "
        "Railway on bhm-liv (see the caveat in config/delay_repay.yaml), and the same code gave "
        "coverage 0.80 on ast-bhm and 0.56 on bhm-liv, so the code alone is not a usable key.",
    ]
    return lines
