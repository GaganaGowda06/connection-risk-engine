import argparse
import sys
import time
from datetime import date
from pathlib import Path

from rail_risk.config import Settings, load_routes
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
    import sqlite3

    import pandas as pd

    from rail_risk.connection import (
        OutboundDelays,
        Train,
        hhmm_to_minutes,
        journey_repay,
        load_stations,
        minutes_to_hhmm,
        service_table,
        simulate,
        to_local_minutes,
    )
    from rail_risk.evaluation import (
        complete_days,
        load_bands,
        load_snapshots,
        load_split_config,
        make_split,
    )
    from rail_risk.models.pipeline import fit_for_inference, load_train_config
    from rail_risk.repay import RepayError, load_rules

    log = get_logger()
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_json)
        log = get_logger()
        cfg = load_train_config(args.config)
        eval_cfg = load_split_config(args.split_file)
        bands = load_bands(args.bands_file)
        station = load_stations(args.stations_file)[args.hub]
        rules = load_rules(args.rules_file)
        planned_min = hhmm_to_minutes(args.outbound_dep)
        snaps = load_snapshots(settings.data_dir / "features" / "snapshots")
    except (ValueError, OSError, KeyError) as exc:
        log.error("connection_invalid", error=str(exc))
        return 1

    # the model is fitted on the train block and calibrated on validation, never on test
    model_snaps = snaps[snaps["route_id"].isin(eval_cfg.routes)] if eval_cfg.routes else snaps
    manifest = settings.data_dir / "interim" / "hsp_manifest.sqlite"
    days = complete_days(
        manifest, sorted(model_snaps["route_id"].unique()), eval_cfg.windows_per_day
    )
    split = make_split([d for d in days if d in set(model_snaps["service_date"])], eval_cfg)
    row = snaps[
        (snaps["rid"] == args.inbound_rid)
        & (snaps["service_date"] == args.date)
        & (snaps["horizon_min"] == args.horizon)
    ]
    if row.empty:
        log.error("connection_invalid", error="no snapshot for that rid, date and horizon")
        return 1
    row = row.iloc[[0]]
    fitted = fit_for_inference(model_snaps, split, bands, cfg)
    levels, quantiles = fitted.quantile.distribution(row)

    # schedule and history: bhm-eus services call at the hub and the destination
    db = sqlite3.connect(f"file:{manifest}?mode=ro", uri=True)
    try:
        rids = {r[0] for r in db.execute("SELECT rid FROM rid_routes WHERE route='bhm-eus'")}
    finally:
        db.close()
    # the unified file is what the feature build reads; per-file schemas differ for all-null columns
    unified = settings.data_dir / "interim" / "_unified" / "hsp_stops.parquet"
    columns = ["rid", "date_of_service", "toc_code", "location", "sched_arr", "sched_dep",
               "actual_arr", "actual_dep"]  # fmt: skip
    all_stops = pd.read_parquet(unified, columns=columns)
    all_stops["_day"] = pd.to_datetime(all_stops["date_of_service"]).dt.date

    def read_days(day_list):
        return all_stops[all_stops["_day"].isin(set(day_list))]

    day_stops = read_days([args.date])
    inbound = day_stops[
        (day_stops["rid"] == args.inbound_rid) & (day_stops["location"] == args.hub)
    ]
    if inbound.empty:
        log.error("connection_invalid", error="inbound service has no stop at the hub that day")
        return 1
    inbound_arr = float(to_local_minutes(inbound["sched_arr"].iloc[[0]], args.date).iloc[0])
    out_today = service_table(day_stops[day_stops["rid"].isin(rids)], args.hub, args.dest)
    out_today = out_today.sort_values("sched_dep")
    hit = out_today[(out_today["sched_dep"] - planned_min).abs() < 0.5]
    if hit.empty:
        log.error("connection_invalid", error="no planned outbound departs at that time")
        return 1
    later = out_today[out_today["sched_dep"] > hit["sched_dep"].iloc[0]].head(3)

    def train_of(r):
        return Train(float(r.sched_dep), float(r.sched_arr), str(r.toc))

    planned = train_of(next(hit.iloc[[0]].itertuples()))
    alternatives = [train_of(r) for r in later.itertuples()]
    history = service_table(
        read_days(split.train)[lambda d: d["rid"].isin(rids)], args.hub, args.dest
    )
    result = simulate(
        levels, quantiles[0], inbound_arr, planned, alternatives, station.minutes,
        OutboundDelays.fit(history), bands.lower_edges, args.draws, cfg["seed"],
    )  # fmt: skip

    h = args.horizon
    print(
        f"inbound {args.inbound_rid} on {args.date}, forecast made T-{h}, scheduled at "
        f"{args.hub} {minutes_to_hhmm(inbound_arr)}"
    )
    print(
        f"outbound {minutes_to_hhmm(planned.sched_dep)} to {args.dest}, planned arrival "
        f"{minutes_to_hhmm(planned.sched_arr)}; alternatives "
        + (", ".join(minutes_to_hhmm(a.sched_dep) for a in alternatives) or "none")
    )
    print(f"minimum connection time at {args.hub}: {station.minutes} min ({station.status})")
    lo_q, med_q, hi_q = 0.1, 0.5, 0.9
    print(f"P(miss): {result.p_miss:.3f}   P(stranded): {result.p_stranded:.3f}")
    if result.stranded:
        print("STRANDED: no draw reaches the destination; no arrival time or payout is given")
    else:
        lo, hi = result.arrival_range(lo_q, hi_q)
        med = planned.sched_arr + result.delay_quantiles[med_q]
        print(
            f"final arrival at {args.dest}: 80% range {minutes_to_hhmm(lo)} to "
            f"{minutes_to_hhmm(hi)}, median {minutes_to_hhmm(med)}"
        )
        print(
            "band probabilities: "
            + ", ".join(
                f"{lab} {p:.3f}" for lab, p in zip(bands.labels, result.band_probs, strict=True)
            )
        )
        toc = args.toc or planned.toc
        try:
            r = journey_repay(result, bands.lower_edges, toc, args.ticket, args.fare, rules)
            print(
                f"expected Delay Repay on a £{args.fare:.2f} {args.ticket} ({toc}): "
                f"£{r.expected_payout:.2f}"
                + ("  [conditional on completing]" if result.p_stranded > 0 else "")
            )
            if r.caveat:
                print(f"note: {r.caveat}")
        except RepayError as exc:
            print(f"expected Delay Repay: not priced ({exc})")
    print(f"{result.n_draws} draws, seed {cfg['seed']}; legs are assumed independent")
    if station.status != "verified":
        print(f"note: the {args.hub} connection time is {station.status}: {station.comment}")
    if args.inbound_rid and set(model_snaps["route_id"].unique()) != {str(row["route_id"].iloc[0])}:
        print(
            f"note: the delay model was fitted on {sorted(model_snaps['route_id'].unique())}, "
            f"not on the inbound route {row['route_id'].iloc[0]}, so its accuracy there is "
            "not validated"
        )
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
    cn.add_argument("--outbound-dep", required=True, help="planned outbound departure HH:MM")
    cn.add_argument("--date", type=date.fromisoformat, required=True)
    cn.add_argument("--fare", type=float, required=True, help="fare in pounds, always supplied")
    cn.add_argument("--ticket", choices=["single", "return"], required=True)
    cn.add_argument("--horizon", type=int, default=60, choices=[120, 60, 15])
    cn.add_argument("--hub", default="BHM")
    cn.add_argument("--dest", default="EUS")
    cn.add_argument("--toc", default=None, help="TOC to price the payout under (default: outbound)")
    cn.add_argument("--draws", type=int, default=2000)
    cn.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    cn.add_argument("--split-file", type=Path, default=Path("config/split.yaml"))
    cn.add_argument("--bands-file", type=Path, default=Path("config/delay_bands.yaml"))
    cn.add_argument("--stations-file", type=Path, default=Path("config/stations.yaml"))
    cn.add_argument("--rules-file", type=Path, default=Path("config/delay_repay.yaml"))
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
    if args.command == "connection":
        return _connection(args)
    if args.command == "evaluate-baselines":
        return _evaluate_baselines(args)
    if args.command == "rebuild-hsp":
        return _rebuild_hsp(args)
    if args.command == "status":
        return _status()
    return 2  # pragma: no cover
