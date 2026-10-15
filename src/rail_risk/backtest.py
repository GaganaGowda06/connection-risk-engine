"""Shared data context for the connection command and the historical connection backtest."""

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rail_risk.connection import (
    OutboundDelays,
    Station,
    TailModel,
    Train,
    inbound_delay_draws,
    load_stations,
    service_table,
    simulate,
    to_local_minutes,
)
from rail_risk.evaluation import (
    Bands,
    SplitConfig,
    complete_days,
    load_bands,
    load_snapshots,
    load_split_config,
    make_split,
    reliability,
)
from rail_risk.features.snapshots import FEATURE_COLUMNS
from rail_risk.lateness import lateness_groups
from rail_risk.models.pipeline import Fitted, fit_for_inference, load_train_config
from rail_risk.repay import DelayRepayRules, load_rules

TRAINED_HORIZONS = (120, 60, 15)
STOP_COLUMNS = [
    "rid", "date_of_service", "toc_code", "location", "sched_arr", "sched_dep", "actual_arr",
    "actual_dep",
]  # fmt: skip


@dataclass
class Context:
    fitted: Fitted
    split: Any
    bands: Bands
    cfg: dict[str, Any]
    eval_cfg: SplitConfig
    station: Station
    rules: DelayRepayRules
    snaps: pd.DataFrame  # every route, for looking up inbound rows
    model_snaps: pd.DataFrame
    all_stops: pd.DataFrame
    eus_rids: set[str]
    ast_rids: set[str]
    history: pd.DataFrame  # bhm-eus services on training dates
    tail: TailModel
    interim: Path
    manifest: Path


def load_context(
    data_dir: Path,
    *,
    config: Path,
    split_file: Path,
    bands_file: Path,
    stations_file: Path,
    rules_file: Path,
    hub: str = "BHM",
    destination: str = "EUS",
) -> Context:
    cfg = load_train_config(config)
    eval_cfg = load_split_config(split_file)
    bands = load_bands(bands_file)
    station = load_stations(stations_file)[hub]
    rules = load_rules(rules_file)
    snaps = load_snapshots(data_dir / "features" / "snapshots")
    model_snaps = snaps[snaps["route_id"].isin(eval_cfg.routes)] if eval_cfg.routes else snaps
    interim = data_dir / "interim"
    manifest = interim / "hsp_manifest.sqlite"
    days = complete_days(
        manifest, sorted(model_snaps["route_id"].unique()), eval_cfg.windows_per_day
    )
    split = make_split([d for d in days if d in set(model_snaps["service_date"])], eval_cfg)
    fitted = fit_for_inference(model_snaps, split, bands, cfg)

    db = sqlite3.connect(f"file:{manifest}?mode=ro", uri=True)
    try:
        eus_rids = {r[0] for r in db.execute("SELECT rid FROM rid_routes WHERE route='bhm-eus'")}
        ast_rids = {r[0] for r in db.execute("SELECT rid FROM rid_routes WHERE route='ast-bhm'")}
    finally:
        db.close()
    # the unified file is what the feature build reads; per-file schemas differ for null columns
    all_stops = pd.read_parquet(interim / "_unified" / "hsp_stops.parquet", columns=STOP_COLUMNS)
    all_stops["_day"] = pd.to_datetime(all_stops["date_of_service"]).dt.date
    train_days = set(split.train)
    history = service_table(
        all_stops[all_stops["_day"].isin(train_days) & all_stops["rid"].isin(eus_rids)],
        hub,
        destination,
    )

    # the tail comes from training rows only
    scored = model_snaps[model_snaps["label_delay_min"].notna()]
    train = scored[scored["service_date"].isin(train_days)]
    levels, q = fitted.quantile.distribution(train)
    tail = TailModel.fit(
        train["label_delay_min"].to_numpy(dtype=float), q[:, -1], lateness_groups(train)
    )
    return Context(
        fitted, split, bands, cfg, eval_cfg, station, rules, snaps, model_snaps, all_stops,
        eus_rids, ast_rids, history, tail, interim, manifest,
    )  # fmt: skip


def day_outbounds(ctx: Context, day: date, hub: str, destination: str) -> pd.DataFrame:
    stops = ctx.all_stops[(ctx.all_stops["_day"] == day) & ctx.all_stops["rid"].isin(ctx.eus_rids)]
    return service_table(stops, hub, destination).sort_values("sched_dep").reset_index(drop=True)


def train_of(row: Any) -> Train:
    return Train(float(row.sched_dep), float(row.sched_arr), str(row.toc))


def exceedance_check(
    ctx: Context, test: pd.DataFrame, threshold: float = 60.0, resamples: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Mean simulated P(delay >= threshold) on the test rows against the observed frequency.

    The predicted side integrates each row's inverse CDF, including the empirical tail, over an
    even grid of probabilities. The observed side has a percentile bootstrap CI over service
    dates. Training rows and offsets only; the test labels are used here for scoring alone.
    """
    scored = test[test["label_delay_min"].notna()]
    levels, q = ctx.fitted.quantile.distribution(scored)
    groups = lateness_groups(scored)
    grid = (np.arange(1000) + 0.5) / 1000
    predicted = np.empty(len(scored))
    for i in range(len(scored)):
        x = inbound_delay_draws(levels, q[i], grid, ctx.tail.for_group(int(groups[i])))
        predicted[i] = float(np.mean(x >= threshold))
    observed = (scored["label_delay_min"].to_numpy(dtype=float) >= threshold).astype(float)
    dates = scored["service_date"].to_numpy()
    udates, inverse = np.unique(dates, return_inverse=True)
    n = np.bincount(inverse).astype(float)
    hits = np.bincount(inverse, weights=observed)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(resamples, len(n)))
    boot = hits[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mean_pred = float(predicted.mean())
    return {
        "threshold": threshold,
        "rows": int(len(scored)),
        "predicted": mean_pred,
        "observed": float(observed.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "within_ci": bool(lo <= mean_pred <= hi),
    }


# backtest


def _nearest_horizon(h: float) -> int:
    return min(TRAINED_HORIZONS, key=lambda t: abs(t - h))


def build_pairs(
    ctx: Context, days: list[date], hub: str, destination: str, mct: float, window: float
) -> pd.DataFrame:
    """Every ast-bhm inbound with each bhm-eus outbound scheduled within the connection window.

    A pair is kept only if outbound departure minus inbound arrival is at least `mct` and at
    most `mct + window`, so no connection that is infeasible on paper is ever scored.
    """
    rows = []
    for day in days:
        day_stops = ctx.all_stops[ctx.all_stops["_day"] == day]
        inbound = day_stops[day_stops["rid"].isin(ctx.ast_rids) & (day_stops["location"] == hub)]
        inbound = inbound.drop_duplicates("rid").dropna(subset=["sched_arr"])
        out = day_outbounds(ctx, day, hub, destination)
        if inbound.empty or out.empty:
            continue
        in_sched = to_local_minutes(inbound["sched_arr"], day).to_numpy()
        in_act = to_local_minutes(
            inbound["actual_arr"].where(inbound["actual_arr"].notna()), day
        ).to_numpy()
        dep_act = (out["sched_dep"] + out["dep_delay"]).to_numpy()
        arr_act = (out["sched_arr"] + out["arr_delay"]).to_numpy()
        for rid, s_arr, a_arr in zip(inbound["rid"], in_sched, in_act, strict=True):
            for i, o in enumerate(out.itertuples()):
                gap = o.sched_dep - s_arr
                if gap < mct or gap > mct + window:
                    continue
                if gap < mct:
                    raise ValueError("infeasible connection reached scoring")
                alts = out.iloc[i + 1 : i + 4]
                missed = (
                    np.nan if np.isnan(a_arr) or np.isnan(dep_act[i])
                    else float(a_arr + mct > dep_act[i])
                )  # fmt: skip
                final = np.nan
                stranded = False
                if missed == 0.0:
                    final = arr_act[i]
                elif missed == 1.0:
                    for j in alts.index:
                        if a_arr + mct <= dep_act[j]:
                            final = arr_act[j]
                            break
                    else:
                        stranded = True
                rows.append(
                    {
                        "date": day, "inbound_rid": rid, "outbound_rid": o.rid,
                        "inbound_sched_arr": float(s_arr), "gap": float(gap),
                        "planned_dep": float(o.sched_dep), "planned_arr": float(o.sched_arr),
                        "alt_index": list(alts.index), "missed": missed,
                        "realised_arrival": final, "stranded": stranded,
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


def snapshots_at(
    ctx: Context, spark: Any, days: list[date], horizons: tuple[int, ...], routes_file: Path
) -> pd.DataFrame:
    """Feature snapshots of ast-bhm services at arbitrary minutes before scheduled arrival."""
    import pyspark.sql.functions as F

    from rail_risk.features.build import load_inputs
    from rail_risk.features.snapshots import build_snapshots

    stops, rid_routes, route_dest = load_inputs(
        spark, ctx.interim, routes_file, min(days), max(days)
    )
    stops = stops.filter(F.col("service_date").isin([d.isoformat() for d in days]))
    rid_routes = rid_routes.filter(F.col("route_id") == "ast-bhm")
    snap = build_snapshots(spark, stops, rid_routes, route_dest, horizons=horizons)
    cols = ["rid", "route_id", "service_date", "horizon_min", *FEATURE_COLUMNS]
    return snap.select(*cols).toPandas()


def run_backtest(
    ctx: Context,
    spark: Any,
    routes_file: Path,
    *,
    hub: str = "BHM",
    destination: str = "EUS",
    window: float = 20.0,
    draws: int = 2000,
    forecast_minutes: tuple[int, ...] = (60, 15),
) -> dict[str, Any]:
    mct = float(ctx.station.minutes)
    db = sqlite3.connect(f"file:{ctx.manifest}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT day FROM windows WHERE route='ast-bhm' GROUP BY day HAVING COUNT(*) >= ?",
            (ctx.eval_cfg.windows_per_day,),
        ).fetchall()
    finally:
        db.close()
    days = sorted(date.fromisoformat(r[0]) for r in rows)
    pairs = build_pairs(ctx, days, hub, destination, mct, window)
    summary: dict[str, Any] = {
        "ast_bhm_days": len(days),
        "min_connection_time": mct,
        "min_connection_time_status": ctx.station.status,
        "window_minutes": window,
        "pairs": int(len(pairs)),
        "pairs_with_known_outcome": int(pairs["missed"].notna().sum()) if len(pairs) else 0,
    }
    if pairs.empty:
        return {**summary, "scores": {}}

    # forecast time is k minutes before the outbound departure, which is h' = k - gap minutes
    # relative to the inbound's scheduled arrival
    need = sorted({int(round(k - g)) for g in pairs["gap"] for k in forecast_minutes})
    snaps = snapshots_at(ctx, spark, days, tuple(need), routes_file)
    snaps["service_date"] = pd.to_datetime(snaps["service_date"].astype(str)).dt.date
    index = {(r.rid, int(r.horizon_min)): i for i, r in enumerate(snaps.itertuples(index=False))}

    bucket = ((pairs["gap"] - mct) // 5).clip(upper=3).astype(int)
    pairs["bucket"] = bucket
    known = pairs[pairs["missed"].notna()]
    out_models: dict[date, OutboundDelays] = {}
    records = []
    for k in forecast_minutes:
        for p in pairs.itertuples():
            h_prime = int(round(k - p.gap))
            at = index.get((p.inbound_rid, h_prime))
            if at is None:
                continue
            feat = snaps.iloc[[at]].copy()
            feat["horizon_min"] = _nearest_horizon(h_prime)
            levels, q = ctx.fitted.quantile.distribution(feat)
            group = int(lateness_groups(feat)[0])
            if p.date not in out_models:  # leave the evaluated date out of the history
                out_models[p.date] = OutboundDelays.fit(
                    ctx.history[ctx.history["date"] != p.date]
                    if (ctx.history["date"] != p.date).any()
                    else ctx.history
                )
            out = day_outbounds(ctx, p.date, hub, destination)
            planned_row = out[(out["sched_dep"] - p.planned_dep).abs() < 0.5].iloc[0]
            alts = [train_of(out.loc[j]) for j in p.alt_index]
            res = simulate(
                levels, q[0], p.inbound_sched_arr, train_of(planned_row), alts, mct,
                out_models[p.date], ctx.bands.lower_edges, draws, ctx.cfg["seed"],
                ctx.tail.for_group(group),
            )  # fmt: skip
            lo, hi = res.arrival_range(0.1, 0.9) or (np.nan, np.nan)
            records.append(
                {"k": k, "pair": p.Index, "p_miss": res.p_miss, "lo": lo, "hi": hi,
                 "p_stranded": res.p_stranded}
            )  # fmt: skip
    pred = pd.DataFrame(records)
    scores: dict[str, Any] = {}
    for k in forecast_minutes:
        sub = pred[pred["k"] == k].merge(
            pairs, left_on="pair", right_index=True, suffixes=("", "_pair")
        )
        sub = sub[sub["missed"].notna()]
        if sub.empty:
            continue
        # baseline: historical miss frequency by scheduled-gap bucket, leaving the date out
        base = []
        for r in sub.itertuples():
            same = known[(known["bucket"] == r.bucket) & (known["date"] != r.date)]
            pool = known[known["date"] != r.date]
            if len(same):
                base.append(float(same["missed"].mean()))
            elif len(pool):
                base.append(float(pool["missed"].mean()))
            else:  # one date only: in-sample, so this is a smoke test, not a score
                base.append(float(known[known["bucket"] == r.bucket]["missed"].mean()))
        sub = sub.assign(baseline=base)
        y = sub["missed"].to_numpy(dtype=float)
        dates = sub["date"].to_numpy()
        bins = int(min(10, max(2, len(sub) // 10)))
        kw = {"bins": bins, "resamples": 1000, "seed": ctx.cfg["seed"]}
        model_r = reliability(dates, sub["p_miss"].to_numpy(), y, **kw)
        base_r = reliability(dates, sub["baseline"].to_numpy(), y, **kw)
        have = sub[sub["realised_arrival"].notna() & sub["lo"].notna()]
        covered = (have["realised_arrival"] >= have["lo"]) & (
            have["realised_arrival"] <= have["hi"]
        )
        scores[f"T-{k} before outbound"] = {
            "pairs": int(len(sub)),
            "miss_rate": float(y.mean()),
            "mean_predicted_miss": float(sub["p_miss"].mean()),
            "model": {"brier": model_r["brier"], "ece": model_r["ece"]},
            "baseline": {"brier": base_r["brier"], "ece": base_r["ece"]},
            "baseline_in_sample": bool(known["date"].nunique() < 2),
            "interval_pairs": int(len(have)),
            "interval_coverage": float(covered.mean()) if len(have) else None,
            "interval_width_min": float((have["hi"] - have["lo"]).mean()) if len(have) else None,
            "stranded_realised": int(sub["stranded"].sum()),
        }
    return {**summary, "scores": scores}
