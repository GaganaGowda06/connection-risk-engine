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
    fit_route_tails,
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
from rail_risk.horizons import HORIZON_GRID
from rail_risk.lateness import lateness_groups
from rail_risk.models.pipeline import Fitted, fit_for_inference, load_train_config
from rail_risk.repay import DelayRepayRules, load_rules

MIN_TRAINED_MINUTES = HORIZON_GRID[0]  # a forecast closer than this is outside the trained grid
TOP_EDGE = 15  # Delay Repay: the first tier
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
    tails: dict[str, TailModel]  # by route, from that route's training rows
    pooled_tail: TailModel
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

    # the tails come from training rows only, one per route and one pooled
    scored = model_snaps[model_snaps["label_delay_min"].notna()]
    train = scored[scored["service_date"].isin(train_days)]
    tails, pooled = fit_route_tails(fitted.quantile, train)
    return Context(
        fitted, split, bands, cfg, eval_cfg, station, rules, snaps, model_snaps, all_stops,
        eus_rids, ast_rids, history, tails, pooled, interim, manifest,
    )  # fmt: skip


def tail_for(ctx: Context, route: str) -> TailModel:
    return ctx.tails.get(route, ctx.pooled_tail)


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
    routes = scored["route_id"].to_numpy()
    grid = (np.arange(1000) + 0.5) / 1000
    predicted = np.empty(len(scored))
    for i in range(len(scored)):
        tail = tail_for(ctx, str(routes[i])).for_group(int(groups[i]))
        x = inbound_delay_draws(levels, q[i], grid, tail)
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
                        "inbound_sched_arr": float(s_arr), "inbound_actual_arr": float(a_arr),
                        "gap": float(gap),
                        "planned_dep": float(o.sched_dep), "planned_arr": float(o.sched_arr),
                        "alt_index": list(alts.index), "missed": missed,
                        "realised_arrival": final, "stranded": stranded,
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


def snapshots_at(
    ctx: Context, spark: Any, days: list[date], horizons: tuple[int, ...], routes_file: Path
) -> pd.DataFrame:
    """Feature snapshots of ast-bhm services at arbitrary minutes before scheduled arrival.

    The historical features come from the full history of every route, not only from `days`, so
    they match the training snapshots. A horizon of zero or less is a forecast made after the
    scheduled arrival.
    """
    import pyspark.sql.functions as F

    from rail_risk.features.build import load_inputs
    from rail_risk.features.snapshots import build_snapshots, destination_services, history_labels

    stops, rid_routes, route_dest = load_inputs(spark, ctx.interim, routes_file)
    services, _ = destination_services(spark, stops, rid_routes, route_dest)
    labels = history_labels(services)
    stops = stops.filter(F.col("service_date").isin([d.isoformat() for d in days]))
    rid_routes = rid_routes.filter(F.col("route_id") == "ast-bhm")
    snap = build_snapshots(spark, stops, rid_routes, route_dest, horizons=horizons, labels=labels)
    cols = ["rid", "route_id", "service_date", "horizon_min", *FEATURE_COLUMNS]
    return snap.select(*cols).toPandas()


def gap_bucket(gap: pd.Series | np.ndarray, mct: float) -> np.ndarray:
    """Scheduled-gap bucket for the baseline: 5 minute steps above the connection time, capped."""
    return np.clip((np.asarray(gap, dtype=float) - mct) // 5, 0, 3).astype(int)


class MissBaseline:
    """Historical miss frequency by scheduled-gap bucket, fitted on training dates only."""

    def __init__(self, train_pairs: pd.DataFrame, mct: float) -> None:
        known = train_pairs[train_pairs["missed"].notna()]
        if known.empty:
            raise ValueError("no training pairs with a known outcome")
        self.mct = mct
        self.overall = float(known["missed"].mean())
        by = known.assign(bucket=gap_bucket(known["gap"], mct)).groupby("bucket")["missed"]
        self.rate = by.mean().to_dict()
        self.count = by.size().to_dict()

    def predict(self, gap: pd.Series | np.ndarray) -> np.ndarray:
        return np.array([self.rate.get(int(b), self.overall) for b in gap_bucket(gap, self.mct)])


def inbound_state(p: Any, k: int) -> tuple[str, int | None]:
    """What is known about the inbound at the forecast time, k minutes before the outbound
    departs: ("arrived", None), or ("approaching" or "due_or_late", exact minutes to its
    scheduled arrival, which is zero or negative once that time has passed)."""
    forecast_at = p.planned_dep - k
    actual = p.inbound_actual_arr
    if not np.isnan(actual) and actual <= forecast_at:
        return "arrived", None
    minutes = int(round(p.inbound_sched_arr - forecast_at))
    return ("approaching" if minutes >= MIN_TRAINED_MINUTES else "due_or_late"), minutes


def _date_ci(
    dates: np.ndarray, hit: np.ndarray, resamples: int, seed: int
) -> tuple[float, float, float]:
    """(mean, ci low, ci high) of a 0/1 outcome with a bootstrap over service dates."""
    udates, inverse = np.unique(dates, return_inverse=True)
    n = np.bincount(inverse).astype(float)
    hits = np.bincount(inverse, weights=hit.astype(float))
    idx = np.random.default_rng(seed).integers(0, len(n), size=(resamples, len(n)))
    boot = hits[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(hit.mean()), float(lo), float(hi)


def score_block(
    sub: pd.DataFrame, bins_cap: int = 10, resamples: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Brier, ECE and reliability table for P(miss) against the baseline, and interval coverage."""
    sub = sub[sub["missed"].notna()]
    y = sub["missed"].to_numpy(dtype=float)
    dates = sub["date"].to_numpy()
    bins = int(min(bins_cap, max(2, len(sub) // 10)))
    kw = {"bins": bins, "resamples": resamples, "seed": seed}
    model_r = reliability(dates, sub["p_miss"].to_numpy(), y, **kw)
    base_r = reliability(dates, sub["baseline"].to_numpy(), y, **kw)
    have = sub[sub["realised_arrival"].notna() & sub["lo"].notna()]
    out: dict[str, Any] = {
        "pairs": int(len(sub)),
        "dates": int(sub["date"].nunique()),
        "miss_rate": float(y.mean()),
        "mean_predicted_miss": float(sub["p_miss"].mean()),
        "model": {"brier": model_r["brier"], "ece": model_r["ece"]},
        "baseline": {"brier": base_r["brier"], "ece": base_r["ece"]},
        "reliability_model": model_r["table"],
        "reliability_baseline": base_r["table"],
        "bins": bins,
        "stranded_realised": int(sub["stranded"].sum()),
        "interval_pairs": int(len(have)),
        "interval_coverage": None,
        "interval_width_min": None,
    }
    if len(have):
        hit = (have["realised_arrival"] >= have["lo"]) & (have["realised_arrival"] <= have["hi"])
        cov, lo, hi = _date_ci(have["date"].to_numpy(), hit.to_numpy(), resamples, seed)
        out["interval_coverage"] = {"value": cov, "ci_low": lo, "ci_high": hi}
        out["interval_width_min"] = float((have["hi"] - have["lo"]).mean())
    return out


def rounding_check(sub: pd.DataFrame, seed: int, resamples: int = 1000) -> dict[str, Any]:
    """Mean P(miss) before and after rounding the inbound draws, against the realised rate.

    The overall figures and the tightest gap bucket, where ties matter most. The realised rate
    has a bootstrap CI over service dates.
    """
    y = sub["missed"].to_numpy(dtype=float)
    rate, lo, hi = _date_ci(sub["date"].to_numpy(), y, resamples, seed)

    def block(m: np.ndarray) -> dict[str, Any]:
        return {
            "pairs": int(m.sum()),
            "realised": float(y[m].mean()),
            "mean_before": float(sub["p_miss_unrounded"].to_numpy()[m].mean()),
            "mean_after": float(sub["p_miss"].to_numpy()[m].mean()),
            "brier_before": float(((sub["p_miss_unrounded"].to_numpy()[m] - y[m]) ** 2).mean()),
            "brier_after": float(((sub["p_miss"].to_numpy()[m] - y[m]) ** 2).mean()),
        }

    everything = np.ones(len(sub), dtype=bool)
    tight = (sub["bucket"] == 0).to_numpy()
    return {
        "all": {**block(everything), "realised_ci_low": lo, "realised_ci_high": hi},
        "tightest_gap_bucket": block(tight) if tight.any() else None,
    }


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
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Out-of-sample connection backtest on ast-bhm inbound against bhm-eus outbound.

    Pairs on test dates are scored; pairs on validation dates are also predicted so that alert
    thresholds can be chosen without touching the test block. The baseline and every fitted part
    (delay model, tail, outbound delays) use training dates only. Each forecast is made k minutes
    before the outbound departs. What the inbound has done by then decides how it is forecast:
    already arrived (its delay is known), still approaching (the model is given the exact
    minutes to its scheduled arrival), or due or late (the scheduled arrival has passed or is
    within the model's shortest horizon; the model sees the true state at that moment and, if
    the train is overdue, the draws are conditioned on it not having arrived yet).
    """
    mct = float(ctx.station.minutes)
    block_of = {d: n for n, days in ctx.split.blocks().items() for d in days}
    db = sqlite3.connect(f"file:{ctx.manifest}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT day FROM windows WHERE route='ast-bhm' GROUP BY day HAVING COUNT(*) >= ?",
            (ctx.eval_cfg.windows_per_day,),
        ).fetchall()
    finally:
        db.close()
    ast_days = {date.fromisoformat(r[0]) for r in rows}
    days = sorted(d for d in ast_days if d in block_of)
    pairs = build_pairs(ctx, days, hub, destination, mct, window)
    pairs["block"] = pairs["date"].map(block_of)
    pairs["bucket"] = gap_bucket(pairs["gap"], mct)
    summary: dict[str, Any] = {
        "ast_bhm_days": {
            n: sum(1 for d in days if block_of[d] == n) for n in ("train", "validation", "test")
        },
        "min_connection_time": mct,
        "min_connection_time_status": ctx.station.status,
        "window_minutes": window,
        "pairs": {n: int((pairs["block"] == n).sum()) for n in ("train", "validation", "test")},
        "pairs_with_known_outcome": {
            n: int(((pairs["block"] == n) & pairs["missed"].notna()).sum())
            for n in ("train", "validation", "test")
        },
        "forecast_minutes": list(forecast_minutes),
        "min_trained_minutes": MIN_TRAINED_MINUTES,
    }
    train_pairs = pairs[pairs["block"] == "train"]
    scored_pairs = pairs[pairs["block"].isin(["validation", "test"])]
    if train_pairs.empty or scored_pairs.empty:
        return {**summary, "scores": {}}, pd.DataFrame()
    baseline = MissBaseline(train_pairs, mct)
    summary["baseline"] = {
        "fitted_on": "training dates only",
        "overall_miss_rate": baseline.overall,
        "rate_by_gap_bucket": {str(k): float(v) for k, v in sorted(baseline.rate.items())},
        "pairs_by_gap_bucket": {str(k): int(v) for k, v in sorted(baseline.count.items())},
    }

    # snapshots needed: one per (inbound, minutes to its scheduled arrival) for pairs not yet in
    plan = []
    for k in forecast_minutes:
        for p in scored_pairs.itertuples():
            state, minutes = inbound_state(p, k)
            plan.append((k, p, state, minutes))
    need = sorted({m for _, _, st, m in plan if st != "arrived"})
    snaps = snapshots_at(
        ctx, spark, sorted(scored_pairs["date"].unique()), tuple(need), routes_file
    )
    index = {(r.rid, int(r.horizon_min)): i for i, r in enumerate(snaps.itertuples(index=False))}

    outbound = OutboundDelays.fit(ctx.history)  # training dates only
    levels = np.asarray(ctx.fitted.quantile.chosen.quantiles)
    tail_pool = tail_for(ctx, "ast-bhm")
    records = []
    for k, p, state, minutes in plan:
        floor = None
        if state == "arrived":
            delay = float(p.inbound_actual_arr - p.inbound_sched_arr)
            q = np.full(len(levels), delay)
            tail = None
        else:
            at = index.get((p.inbound_rid, minutes))
            if at is None:
                continue
            feat = snaps.iloc[[at]].copy()
            _, q_all = ctx.fitted.quantile.distribution(feat)
            q = q_all[0]
            tail = tail_pool.for_group(int(lateness_groups(feat)[0]))
            if minutes < 0:
                floor = float(-minutes)
        out = day_outbounds(ctx, p.date, hub, destination)
        planned_row = out[(out["sched_dep"] - p.planned_dep).abs() < 0.5].iloc[0]
        alts = [train_of(out.loc[j]) for j in p.alt_index]
        res = simulate(
            levels, q, p.inbound_sched_arr, train_of(planned_row), alts, mct, outbound,
            ctx.bands.lower_edges, draws, ctx.cfg["seed"], tail, floor,
        )  # fmt: skip
        before = simulate(
            levels, q, p.inbound_sched_arr, train_of(planned_row), alts, mct, outbound,
            ctx.bands.lower_edges, draws, ctx.cfg["seed"], tail, floor, round_inbound=False,
        )  # fmt: skip
        lo, hi = res.arrival_range(0.1, 0.9) or (np.nan, np.nan)
        final_delay = (
            np.nan if np.isnan(p.realised_arrival) else float(p.realised_arrival - p.planned_arr)
        )
        record = {
            "block": p.block, "date": p.date, "k": k, "pair": p.Index, "state": state,
            "minutes_to_arrival": minutes, "inbound_rid": p.inbound_rid,
            "outbound_rid": p.outbound_rid, "gap": p.gap, "bucket": p.bucket,
            "p_miss": res.p_miss, "p_miss_unrounded": before.p_miss,
            "p_stranded": res.p_stranded,
            "lo": lo, "hi": hi, "missed": p.missed, "stranded": p.stranded,
            "realised_arrival": p.realised_arrival, "final_delay": final_delay,
            "baseline": float(baseline.predict([p.gap])[0]),
        }  # fmt: skip
        records.append(record)
    pred = pd.DataFrame(records)
    scores: dict[str, Any] = {}
    for block in ("test", "validation"):
        for k in forecast_minutes:
            sub = pred[(pred["block"] == block) & (pred["k"] == k)]
            sub = sub[sub["missed"].notna()]
            if sub.empty:
                continue
            cell = score_block(sub, seed=ctx.cfg["seed"])
            cell["by_state"] = {
                st: {
                    "pairs": int(len(g)),
                    "miss_rate": float(g["missed"].mean()),
                    "mean_predicted_miss": float(g["p_miss"].mean()),
                    "brier_model": float(((g["p_miss"] - g["missed"]) ** 2).mean()),
                    "brier_baseline": float(((g["baseline"] - g["missed"]) ** 2).mean()),
                }
                for st, g in sub.groupby("state")
            }
            # the subset whose inbound had not arrived when the forecast was made: a fact known
            # at prediction time, not chosen by the outcome
            waiting = sub[sub["state"] != "arrived"]
            cell["not_yet_arrived"] = (
                {
                    **score_block(waiting, seed=ctx.cfg["seed"]),
                    "excluded_arrived": int(len(sub) - len(waiting)),
                }
                if len(waiting) >= 20
                else None
            )
            cell["rounding_check"] = rounding_check(sub, ctx.cfg["seed"])
            scores.setdefault(block, {})[f"T-{k} before outbound"] = cell
    return {**summary, "scores": scores}, pred
