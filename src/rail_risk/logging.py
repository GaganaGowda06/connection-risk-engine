import logging
import sys

import structlog

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure structlog with ISO UTC timestamps; JSON output when json is true."""
    name = level.upper()
    if name not in LEVELS:
        raise ValueError(f"unknown log level {level!r}; expected one of {', '.join(LEVELS)}")
    renderer = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, name)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = "rail_risk"):
    return structlog.get_logger(name)
