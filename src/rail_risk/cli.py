import argparse
import sys
import time
from datetime import date
from pathlib import Path

from rail_risk.config import Settings, load_routes
from rail_risk.horizons import HORIZON_GRID
from rail_risk.logging import configure_logging, get_logger


def _check() -> int:
    configure_logging()
    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
        routes = load_routes(settings.routes_file)
    except (ValueError, OSError) as exc:
        log.error("config_invalid", error=str(exc))
        return 1
    log.info(
        "config_ok",
        routes=[r.id for r in routes.routes],
        darwin_credentials_set=(
            settings.rdm_darwin_user is not None and settings.rdm_darwin_password is not None
        ),
        hsp_key_set=settings.rdm_hsp_key is not None,
    )
    return 0


def _hours(text: str) -> range:
    try:
        start, end = (int(part) for part in text.split("-"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hours must look like 06-22") from exc
    if not 0 <= start < end <= 24:
        raise argparse.ArgumentTypeError("hours must satisfy 0 <= start < end <= 24")
    return range(start, end)


def _ingest_hsp(args: argparse.Namespace) -> int:
    from rail_risk.ingest.hsp import HspClient, HspHttpError, Manifest, backfill

    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
        routes = load_routes(settings.routes_file).routes
        route = next((r for r in routes if r.id == args.route), None)
        if route is None:
            raise ValueError(f"unknown route {args.route!r}; known: {[r.id for r in routes]}")
        if settings.rdm_hsp_key is None:
            raise ValueError("RAIL_RDM_HSP_KEY is not set")
        if args.end < args.start:
            raise ValueError("--end must not be before --start")
    except (ValueError, OSError) as exc:
        log.error("config_invalid", error=str(exc))
        return 1
    raw_root = settings.data_dir / "raw" / "hsp"
    interim = settings.data_dir / "interim"
    client = HspClient(settings.rdm_hsp_key.get_secret_value())
    manifest = Manifest(interim / "hsp_manifest.sqlite")
    started = time.monotonic()
    try:
        c = backfill(
            route,
            args.start,
            args.end,
            args.hours,
            client,
            manifest,
            raw_root=raw_root,
            interim_root=interim,
            newest_first=args.newest_first,
        )
    except HspHttpError as exc:
        log.error("ingest_aborted", error=str(exc))
        return 1
    finally:
        manifest.close()
    log.info(
        "ingest_summary",
        route=route.id,
        seconds=round(time.monotonic() - started),
        metrics_calls=c.metrics_calls,
        detail_calls=c.detail_calls,
        retries=c.retries,
        windows_done=c.windows_done,
        windows_skipped=c.windows_skipped,
        failed_windows=c.failed_windows,
        services=c.services,
        rows=c.rows,
        quarantined_rows=c.quarantined_rows,
        empty_details=c.empty_details,
        quarantine_reasons=c.reasons,
    )
    # 2 means some windows failed and a rerun will retry them; 1 means stop and look
    return 2 if c.failed_windows else 0


def _rebuild_hsp(args: argparse.Namespace) -> int:
    from rail_risk.ingest.hsp import Manifest, rebuild_interim

    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
    except ValueError as exc:
        log.error("config_invalid", error=str(exc))
        return 1
    interim = settings.data_dir / "interim"
    manifest = Manifest(interim / "hsp_manifest.sqlite")
    try:
        rows, bad = rebuild_interim(manifest, settings.data_dir / "raw" / "hsp", interim)
    finally:
        manifest.close()
    log.info("rebuild_done", rows=rows, quarantined_rows=bad)
    return 0


def _parse_darwin(args: argparse.Namespace) -> int:
    import os

    from rail_risk.ingest.darwin import DarwinManifest, parse_directory

    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
    except ValueError as exc:
        log.error("config_invalid", error=str(exc))
        return 1
    os.nice(10)  # the capture and the backfill are running; stay out of their way
    interim = settings.data_dir / "interim"
    manifest = DarwinManifest(interim / "darwin_manifest.sqlite")
    try:
        results = parse_directory(
            settings.data_dir / "raw" / "darwin", interim, manifest, only_date=args.date
        )
    finally:
        manifest.close()
    log.info(
        "parse_summary",
        files=len(results),
        seconds=round(sum(r.seconds for r in results), 1),
        bytes_lost=sum(r.stats.bytes_lost for r in results),
        duplicates=sum(r.stats.duplicates for r in results),
    )
    return 0


def _build_features(args: argparse.Namespace) -> int:
    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
        if args.start and args.end and args.end < args.start:
            raise ValueError("--end must not be before --start")
    except ValueError as exc:
        log.error("config_invalid", error=str(exc))
        return 1
    if not args.skip_darwin:
        rc = _parse_darwin(argparse.Namespace(date=None))
        if rc:
            return rc
    from rail_risk.features.build import build
    from rail_risk.spark import build_spark

    spark = build_spark()
    stats = build(
        spark,
        settings.data_dir / "interim",
        settings.data_dir / "features",
        settings.routes_file,
        args.start,
        args.end,
    )
    print(
        f"rows={stats.rows} per_horizon={stats.rows_per_horizon} "
        f"label_null_rate={stats.label_null_rate:.3f} seconds={stats.seconds:.0f} "
        f"peak_mb={stats.peak_mb:.0f}"
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    from rail_risk.evaluation import (
        complete_days,
        load_bands,
        load_snapshots,
        load_split_config,
        make_split,
    )
    from rail_risk.models.pipeline import (
        git_state,
        guard,
        load_train_config,
        log_to_mlflow,
        run_training,
        write_model_report,
    )

    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
        cfg = load_train_config(args.config)
        eval_cfg = load_split_config(args.split_file)
        bands = load_bands(args.bands_file)
        snaps = load_snapshots(settings.data_dir / "features" / "snapshots")
        if eval_cfg.routes:
            snaps = snaps[snaps["route_id"].isin(eval_cfg.routes)]
        routes = sorted(snaps["route_id"].unique())
        days = complete_days(
            settings.data_dir / "interim" / "hsp_manifest.sqlite", routes, eval_cfg.windows_per_day
        )
        split = make_split([d for d in days if d in set(snaps["service_date"])], eval_cfg)
    except (ValueError, OSError) as exc:
        log.error("train_invalid", error=str(exc))
        return 1
    sha, dirty = git_state()
    tag = guard(len(split.train), cfg["min_train_days"], dirty)
    if dirty and len(split.train) >= cfg["min_train_days"]:
        log.warning(
            "tree_dirty_not_reportable",
            commit=sha,
            hint="commit and push first so the logged sha describes the code that ran",
        )
    result, fitted = run_training(snaps, split, bands, eval_cfg, cfg)
    run_id = log_to_mlflow(
        result,
        fitted,
        tracking_dir=Path("mlruns"),
        tag=tag,
        snapshot_path=settings.data_dir / "features" / "snapshots",
        train_range=(split.train[0].isoformat(), split.train[-1].isoformat()),
        seed=cfg["seed"],
        git_sha=sha,
        git_dirty=dirty,
    )
    if tag == "reportable":
        write_model_report(result, args.report, run_id, sha)
    log.info(
        "train_done",
        run_id=run_id,
        tag=tag,
        commit=sha,
        train_days=len(split.train),
        min_train_days=cfg["min_train_days"],
        report=str(args.report) if tag == "reportable" else "not written (smoke run)",
    )
    return 0


def _connection(args: argparse.Namespace) -> int:
    from rail_risk.backtest import day_outbounds, load_context, train_of
    from rail_risk.connection import (
        InfeasibleConnection,
        choose_outbound,
        hhmm_to_minutes,
        journey_repay,
        minutes_to_hhmm,
        simulate,
        to_local_minutes,
    )
    from rail_risk.evaluation import load_bands  # noqa: F401
    from rail_risk.lateness import lateness_groups
    from rail_risk.repay import RepayError

    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
        planned_min = hhmm_to_minutes(args.outbound_dep) if args.outbound_dep else None
        ctx = load_context(
            settings.data_dir,
            config=args.config,
            split_file=args.split_file,
            bands_file=args.bands_file,
            stations_file=args.stations_file,
            rules_file=args.rules_file,
            hub=args.hub,
            destination=args.dest,
        )
    except (ValueError, OSError, KeyError) as exc:
        log.error("connection_invalid", error=str(exc))
        return 1

    station, bands = ctx.station, ctx.bands
    row = ctx.snaps[
        (ctx.snaps["rid"] == args.inbound_rid)
        & (ctx.snaps["service_date"] == args.date)
        & (ctx.snaps["horizon_min"] == args.horizon)
    ]
    if row.empty:
        log.error("connection_invalid", error="no snapshot for that rid, date and horizon")
        return 1
    row = row.iloc[[0]]
    levels, quantiles = ctx.fitted.quantile.distribution(row)
    day_stops = ctx.all_stops[ctx.all_stops["_day"] == args.date]
    inbound = day_stops[
        (day_stops["rid"] == args.inbound_rid) & (day_stops["location"] == args.hub)
    ]
    if inbound.empty:
        log.error("connection_invalid", error="inbound service has no stop at the hub that day")
        return 1
    inbound_arr = float(to_local_minutes(inbound["sched_arr"].iloc[[0]], args.date).iloc[0])
    table = day_outbounds(ctx, args.date, args.hub, args.dest)
    trains = [train_of(r) for r in table.itertuples()]
    try:
        planned = choose_outbound(inbound_arr, trains, station.minutes, planned_min)
    except InfeasibleConnection as exc:
        print(f"refused: {exc}")
        return 1
    chosen_note = "" if planned_min is not None else "  (first feasible outbound, none given)"
    later = [t for t in trains if t.sched_dep > planned.sched_dep][:3]
    group = int(lateness_groups(row)[0])
    from rail_risk.backtest import tail_for

    tail = tail_for(ctx, str(row["route_id"].iloc[0])).for_group(group)
    from rail_risk.connection import OutboundDelays

    outbound = OutboundDelays.fit(ctx.history)
    toc = args.toc or planned.toc
    stranded_note = (
        "stranded draws (no catchable train that day) are priced in the top tier, an assumption "
        "that treats a stranded passenger as claiming the longest delay"
    )

    def run(mct: float):
        return simulate(
            levels, quantiles[0], inbound_arr, planned, later, mct, outbound,
            bands.lower_edges, args.draws, ctx.cfg["seed"], tail,
        )  # fmt: skip

    def payout(result) -> str:
        try:
            r = journey_repay(result, bands.lower_edges, toc, args.ticket, args.fare, ctx.rules)
        except RepayError as exc:
            return f"not priced ({exc})"
        return f"£{r.expected_payout:.2f}"

    def arrival(result) -> str:
        rng = result.arrival_range(0.1, 0.9)
        if rng is None:
            return "stranded, no arrival"
        return f"{minutes_to_hhmm(rng[0])} to {minutes_to_hhmm(rng[1])}"

    h = args.horizon
    gap = planned.sched_dep - inbound_arr
    print(
        f"inbound {args.inbound_rid} on {args.date}, forecast made T-{h}, due at {args.hub} "
        f"{minutes_to_hhmm(inbound_arr)}"
    )
    print(
        f"outbound {minutes_to_hhmm(planned.sched_dep)} to {args.dest}, planned arrival "
        f"{minutes_to_hhmm(planned.sched_arr)}, scheduled gap {gap:.0f} min{chosen_note}; "
        f"alternatives " + (", ".join(minutes_to_hhmm(a.sched_dep) for a in later) or "none")
    )
    print(f"minimum connection time at {args.hub}: {station.minutes} min ({station.status})")
    base = run(station.minutes)
    print(f"P(miss): {base.p_miss:.3f}   P(stranded): {base.p_stranded:.3f}")
    if base.stranded:
        print("STRANDED: no draw reaches the destination; no arrival time is given")
    else:
        med = planned.sched_arr + base.delay_quantiles[0.5]
        print(
            f"final arrival at {args.dest}: 80% range {arrival(base)}, median "
            f"{minutes_to_hhmm(med)}"
            + ("  (range excludes stranded draws)" if base.p_stranded > 0 else "")
        )
    print(
        "band probabilities: "
        + ", ".join(f"{lab} {p:.3f}" for lab, p in zip(bands.labels, base.band_probs, strict=True))
    )
    print(f"expected Delay Repay on a £{args.fare:.2f} {args.ticket} ({toc}): {payout(base)}")
    print(f"  {stranded_note}")
    print(
        f"sensitivity to the {args.hub} minimum connection time (the {station.minutes} min "
        f"figure is {station.status}):"
    )
    print("  mct   P(miss)   80% arrival range   expected payout")
    for mct in (5, 7, 10, 15):
        if gap < mct:
            print(f"  {mct:>3}   infeasible on paper at this time (gap {gap:.0f} min), not scored")
            continue
        r = run(float(mct))
        print(f"  {mct:>3}   {r.p_miss:>7.3f}   {arrival(r):<17}   {payout(r)}")
    print(f"{base.n_draws} draws, seed {ctx.cfg['seed']}; legs are assumed independent")
    if station.status != "verified":
        print(f"note: the {args.hub} connection time is {station.status}: {station.comment}")
    try:
        caveat = ctx.rules.operator(toc).caveat
    except RepayError:
        caveat = None
    if caveat:
        print(f"note: {caveat}")
    fitted_on = sorted(ctx.model_snaps["route_id"].unique())
    if str(row["route_id"].iloc[0]) not in fitted_on:
        print(
            f"note: the delay model was fitted on {fitted_on}, not on the inbound route "
            f"{row['route_id'].iloc[0]}, so its accuracy there is not validated"
        )
    return 0


def _backtest(args: argparse.Namespace) -> int:
    import json

    from rail_risk.backtest import load_context, run_backtest
    from rail_risk.spark import build_spark

    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
        ctx = load_context(
            settings.data_dir,
            config=args.config,
            split_file=args.split_file,
            bands_file=args.bands_file,
            stations_file=args.stations_file,
            rules_file=args.rules_file,
        )
    except (ValueError, OSError, KeyError) as exc:
        log.error("backtest_invalid", error=str(exc))
        return 1
    from rail_risk.alerts import alert_report, load_alert_config, policy_stats
    from rail_risk.connection_report import alert_inputs, write_connections_section

    result, pred = run_backtest(
        ctx, build_spark(), settings.routes_file, window=args.window, draws=args.draws
    )
    if not pred.empty:
        pred_path = settings.data_dir / "features" / "connection_predictions.parquet"
        pred.to_parquet(pred_path)  # data/ is git-ignored; kept for inspecting the scores
        acfg = load_alert_config(args.alerts_file)
        alerts = {}
        for name, column, outcome in (
            ("likely_to_miss", "p_miss", "missed"),
            ("may_be_owed_delay_repay", "p_final_ge15", "delay15"),
        ):
            (pv, yv), (pt, yt), per_k = alert_inputs(pred, column, outcome)
            report = alert_report(name, (pv, yv), (pt, yt), acfg)
            report["test_by_forecast"] = {
                k: policy_stats(p, y, report["tau"], acfg.false_alert, acfg.missed_alert)
                for k, (p, y) in per_k.items()
            }
            alerts[name] = report
        result["alerts"] = alerts
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    if args.report and not pred.empty:
        write_connections_section(result, args.report)
    print(f"ast-bhm days: {result['ast_bhm_days']}; pairs: {result['pairs']}")
    print(
        f"minimum connection time {result['min_connection_time']:.0f} min "
        f"({result['min_connection_time_status']})"
    )
    for block, per_k in result["scores"].items():
        for name, sc in per_k.items():
            m, b = sc["model"], sc["baseline"]
            print(
                f"[{block}] {name}: {sc['pairs']} pairs, miss rate {sc['miss_rate']:.3f}, "
                f"mean predicted {sc['mean_predicted_miss']:.3f}; "
                f"model Brier {m['brier']['value']:.4f} ECE {m['ece']['value']:.4f} vs "
                f"baseline Brier {b['brier']['value']:.4f} ECE {b['ece']['value']:.4f}"
            )
            if sc["interval_coverage"] is not None:
                c = sc["interval_coverage"]
                print(
                    f"  80% arrival range covered {c['value']:.3f} "
                    f"({c['ci_low']:.3f} to {c['ci_high']:.3f}) of {sc['interval_pairs']} pairs, "
                    f"mean width {sc['interval_width_min']:.1f} min"
                )
    for name, a in result.get("alerts", {}).items():
        t = a["test"]["chosen"]
        print(f"{name}: tau {a['tau']:.2f}, test {t}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    configure_logging(Settings().log_level, Settings().log_json)
    uvicorn.run(
        "rail_risk.api.app:app_from_environment", factory=True, host=args.host,
        port=args.port, log_level="warning",
    )  # fmt: skip
    return 0


def _drift(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    import pandas as pd

    from rail_risk.drift import drift_report, write_report
    from rail_risk.evaluation import complete_days, load_snapshots, load_split_config, make_split
    from rail_risk.features.snapshots import CONTEXT_FEATURES, HISTORY_FEATURES, LIVE_FEATURES
    from rail_risk.live import FeatureService, LiveStore, load_tiplocs, resolve_route
    from rail_risk.spark import build_spark

    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
        eval_cfg = load_split_config(args.split_file)
        snaps = load_snapshots(settings.data_dir / "features" / "snapshots")
        if eval_cfg.routes:
            snaps = snaps[snaps["route_id"].isin(eval_cfg.routes)]
        days = complete_days(
            settings.data_dir / "interim" / "hsp_manifest.sqlite",
            sorted(snaps["route_id"].unique()),
            eval_cfg.windows_per_day,
        )
        split = make_split([d for d in days if d in set(snaps["service_date"])], eval_cfg)
        routes = load_routes(settings.routes_file).routes
    except (ValueError, OSError) as exc:
        log.error("drift_invalid", error=str(exc))
        return 1
    train = snaps[
        snaps["service_date"].isin(set(split.train))
        & (snaps["horizon_min"] == args.minutes)
        & snaps["label_delay_min"].notna()
    ]

    store = LiveStore(
        settings.data_dir / "interim", settings.data_dir / "raw" / "darwin", load_tiplocs(),
        days_back=args.days,
    )  # fmt: skip
    store.refresh()
    now = pd.Timestamp(datetime.now(UTC))
    dest_tiplocs = {t for t, crs in store.tiplocs.items() if crs in {r.destination for r in routes}}
    ev = store.events
    arrived = ev[
        (ev["event_type"] == "arr")
        & ev["tiploc"].isin(dest_tiplocs)
        & (ev["planned_time"] <= now)
        & (ev["planned_time"] >= now - pd.Timedelta(days=args.days))
    ]
    batch = {}
    for rid in sorted(set(arrived["rid"])):
        table = store.stop_table(rid)
        route_id = resolve_route(table, routes) if len(table) >= 2 else None
        if route_id is None or route_id not in set(snaps["route_id"]):
            continue
        batch[rid] = (table, route_id, args.minutes)
    features = FeatureService(
        build_spark("drift", cores=2, driver_memory="2g", shuffle_partitions=2, adaptive=False),
        settings.data_dir / "interim",
        settings.routes_file,
    )
    rows = []
    items = list(batch.items())
    for start in range(0, len(items), 150):
        rows.extend(features.batch(dict(items[start : start + 150])).values())
    live = pd.DataFrame(rows)
    names = [*LIVE_FEATURES, *CONTEXT_FEATURES, *HISTORY_FEATURES]
    categorical = ["toc_code"]
    numeric = [c for c in names if c not in categorical]
    first = arrived["planned_time"].min()
    report = drift_report(
        train, live, numeric, categorical,
        window={
            "live_from": None if pd.isna(first) else first.date().isoformat(),
            "live_to": now.date().isoformat(),
            "minutes": args.minutes,
            "days_requested": args.days,
            "caveat": (
                "Darwin capture started on 2026-10-15, so the live window holds fewer than "
                f"{args.days} days of services. Daily and weekly patterns in the live sample are "
                "partial, and PSI on a short window is noisy."
            ),
        },
    )  # fmt: skip
    write_report(report, args.out)
    top = report["features"][:3]
    print(
        f"drift: {report['live_rows']} live services against {report['train_rows']} training rows; "
        f"summary {report['summary']}"
    )
    for r in top:
        print(f"  {r['feature']}: PSI {r['psi']:.3f} ({r['status']})")
    return 0


def _ui(args: argparse.Namespace) -> int:
    import subprocess
    import sys

    path = Path(__file__).parent / "ui" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(path),
        "--server.address", args.host, "--server.port", str(args.port),
        "--server.headless", "true", "--browser.gatherUsageStats", "false",
        "--client.toolbarMode", "minimal",
    ]  # fmt: skip
    return subprocess.run(cmd, check=False).returncode  # noqa: S603


def _promote(args: argparse.Namespace) -> int:
    from rail_risk.models.registry import register_champion

    log = get_logger()
    try:
        version = register_champion(args.mlruns, args.run_id, alias=args.alias)
    except (ValueError, OSError) as exc:
        log.error("promote_failed", error=str(exc))
        return 1
    print(f"registered run {args.run_id} as connection-risk version {version}, alias {args.alias}")
    return 0


def _evaluate_baselines(args: argparse.Namespace) -> int:
    from rail_risk.evaluation import (
        complete_days,
        evaluate,
        load_bands,
        load_snapshots,
        load_split_config,
        make_split,
        write_reports,
    )

    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
        cfg = load_split_config(args.split_file)
        bands = load_bands(args.bands_file)
        snaps = load_snapshots(settings.data_dir / "features" / "snapshots")
        if cfg.routes:
            snaps = snaps[snaps["route_id"].isin(cfg.routes)]
        routes = sorted(snaps["route_id"].unique())
        days = complete_days(
            settings.data_dir / "interim" / "hsp_manifest.sqlite", routes, cfg.windows_per_day
        )
        split = make_split([d for d in days if d in set(snaps["service_date"])], cfg)
    except (ValueError, OSError) as exc:
        log.error("evaluation_invalid", error=str(exc))
        return 1
    result = evaluate(snaps, split, bands, cfg)
    write_reports(result, args.out)
    log.info(
        "baselines_scored",
        train_days=len(split.train),
        validation_days=len(split.validation),
        test_days=len(split.test),
        out=str(args.out),
    )
    return 0


def _status() -> int:
    from rail_risk.status import darwin_status, hsp_status

    try:
        data_dir = Settings().data_dir
    except ValueError as exc:
        print(f"config_invalid: {exc}")
        return 1
    print("HSP")
    for line in hsp_status(data_dir / "interim" / "hsp_manifest.sqlite"):
        print(f"  {line}")
    print("Darwin")
    for line in darwin_status(data_dir / "raw" / "darwin", data_dir / "interim"):
        print(f"  {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rail-risk")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate settings and routes")
    ing = sub.add_parser("ingest-hsp", help="resumable HSP backfill for one route")
    ing.add_argument("--route", required=True)
    ing.add_argument("--start", type=date.fromisoformat, required=True)
    ing.add_argument("--end", type=date.fromisoformat, required=True)
    ing.add_argument("--hours", type=_hours, default=range(6, 22))
    ing.add_argument("--newest-first", action="store_true")
    pd_ = sub.add_parser("parse-darwin", help="parse closed Darwin hourly files into Parquet")
    pd_.add_argument("--date", type=date.fromisoformat, default=None)
    bf = sub.add_parser("build-features", help="parse Darwin, then build feature snapshots")
    bf.add_argument("--start", type=date.fromisoformat, default=None)
    bf.add_argument("--end", type=date.fromisoformat, default=None)
    bf.add_argument("--skip-darwin", action="store_true")
    tr = sub.add_parser("train", help="fit quantile and band models, log to MLflow")
    tr.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    tr.add_argument("--split-file", type=Path, default=Path("config/split.yaml"))
    tr.add_argument("--bands-file", type=Path, default=Path("config/delay_bands.yaml"))
    tr.add_argument("--report", type=Path, default=Path("reports/model_results.md"))
    ev = sub.add_parser("evaluate-baselines", help="score the baselines on validation and test")
    ev.add_argument("--split-file", type=Path, default=Path("config/split.yaml"))
    ev.add_argument("--bands-file", type=Path, default=Path("config/delay_bands.yaml"))
    ev.add_argument("--out", type=Path, default=Path("reports"))
    cn = sub.add_parser("connection", help="miss probability and expected payout for a connection")
    cn.add_argument("--inbound-rid", required=True)
    cn.add_argument(
        "--outbound-dep",
        default=None,
        help="planned outbound departure HH:MM; omitted: the first feasible one",
    )
    cn.add_argument("--date", type=date.fromisoformat, required=True)
    cn.add_argument("--fare", type=float, required=True, help="fare in pounds, always supplied")
    cn.add_argument("--ticket", choices=["single", "return"], required=True)
    cn.add_argument("--horizon", type=int, default=60, choices=list(HORIZON_GRID))
    cn.add_argument("--hub", default="BHM")
    cn.add_argument("--dest", default="EUS")
    cn.add_argument("--toc", default=None, help="TOC to price the payout under (default: outbound)")
    cn.add_argument("--draws", type=int, default=2000)
    cn.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    cn.add_argument("--split-file", type=Path, default=Path("config/split.yaml"))
    cn.add_argument("--bands-file", type=Path, default=Path("config/delay_bands.yaml"))
    cn.add_argument("--stations-file", type=Path, default=Path("config/stations.yaml"))
    cn.add_argument("--rules-file", type=Path, default=Path("config/delay_repay.yaml"))
    sv = sub.add_parser("serve", help="run the risk API on localhost")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="127.0.0.1", help="0.0.0.0 only inside a container")
    dr = sub.add_parser("drift", help="PSI of live serving features against training")
    dr.add_argument("--days", type=int, default=7)
    dr.add_argument(
        "--minutes", type=int, default=60, help="features built this long before arrival"
    )
    dr.add_argument("--split-file", type=Path, default=Path("config/split.yaml"))
    dr.add_argument("--out", type=Path, default=Path("reports"))
    ui = sub.add_parser("ui", help="run the Streamlit page on localhost (needs `serve`)")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--host", default="127.0.0.1")
    pr = sub.add_parser("promote", help="register a reportable run and point an alias at it")
    pr.add_argument("--run-id", required=True)
    pr.add_argument("--alias", default="champion")
    pr.add_argument("--mlruns", type=Path, default=Path("mlruns"))
    bt = sub.add_parser("backtest-connections", help="score connection predictions on history")
    bt.add_argument("--window", type=float, default=20.0, help="minutes beyond the minimum time")
    bt.add_argument("--draws", type=int, default=2000)
    bt.add_argument("--out", type=Path, default=Path("reports/connection_backtest.json"))
    bt.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    bt.add_argument("--split-file", type=Path, default=Path("config/split.yaml"))
    bt.add_argument("--bands-file", type=Path, default=Path("config/delay_bands.yaml"))
    bt.add_argument("--stations-file", type=Path, default=Path("config/stations.yaml"))
    bt.add_argument("--rules-file", type=Path, default=Path("config/delay_repay.yaml"))
    bt.add_argument("--alerts-file", type=Path, default=Path("config/alerts.yaml"))
    bt.add_argument("--report", type=Path, default=Path("reports/model_results.md"))
    sub.add_parser("rebuild-hsp", help="re-parse saved raw HSP details into interim Parquet")
    sub.add_parser("status", help="show HSP backfill and Darwin capture progress")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "check":
        return _check()
    if args.command == "ingest-hsp":
        return _ingest_hsp(args)
    if args.command == "parse-darwin":
        return _parse_darwin(args)
    if args.command == "build-features":
        return _build_features(args)
    if args.command == "train":
        return _train(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "drift":
        return _drift(args)
    if args.command == "ui":
        return _ui(args)
    if args.command == "promote":
        return _promote(args)
    if args.command == "backtest-connections":
        return _backtest(args)
    if args.command == "connection":
        return _connection(args)
    if args.command == "evaluate-baselines":
        return _evaluate_baselines(args)
    if args.command == "rebuild-hsp":
        return _rebuild_hsp(args)
    if args.command == "status":
        return _status()
    return 2  # pragma: no cover
