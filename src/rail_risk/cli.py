import argparse
import sys

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
        darwin_key_set=settings.rdm_darwin_key is not None,
        hsp_key_set=settings.rdm_hsp_key is not None,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rail-risk")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate settings and routes")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "check":
        return _check()
    return 2  # pragma: no cover
