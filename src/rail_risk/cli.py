import argparse
import sys
import time
from datetime import date

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
    return 1 if c.failed_windows else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rail-risk")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate settings and routes")
    ing = sub.add_parser("ingest-hsp", help="resumable HSP backfill for one route")
    ing.add_argument("--route", required=True)
    ing.add_argument("--start", type=date.fromisoformat, required=True)
    ing.add_argument("--end", type=date.fromisoformat, required=True)
    ing.add_argument("--hours", type=_hours, default=range(6, 22))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "check":
        return _check()
    if args.command == "ingest-hsp":
        return _ingest_hsp(args)
    return 2  # pragma: no cover
