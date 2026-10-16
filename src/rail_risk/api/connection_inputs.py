"""What /connection needs besides the delay model: station times, the alert threshold, the
outbound delay history and the tail model. Each part is optional, and a missing part is logged
and shown in the response rather than hidden."""

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from rail_risk.alerts import load_alert_config
from rail_risk.connection import OutboundDelays, fit_route_tails, load_stations, service_table
from rail_risk.logging import get_logger

HUB, FINAL, OUTBOUND_ROUTE = "BHM", "EUS", "bhm-eus"
STOP_COLUMNS = [
    "rid", "date_of_service", "toc_code", "location", "sched_arr", "sched_dep", "actual_arr",
    "actual_dep",
]  # fmt: skip


def outbound_history(interim: Path) -> OutboundDelays:
    """Past BHM to EUS services, every day on disk, for the outbound delay draws."""
    stops = pd.read_parquet(interim / "_unified" / "hsp_stops.parquet", columns=STOP_COLUMNS)
    db = sqlite3.connect(f"file:{interim / 'hsp_manifest.sqlite'}?mode=ro", uri=True)
    try:
        rids = {
            r[0] for r in db.execute("SELECT rid FROM rid_routes WHERE route=?", (OUTBOUND_ROUTE,))
        }
    finally:
        db.close()
    return OutboundDelays.fit(service_table(stops[stops["rid"].isin(rids)], HUB, FINAL))


def route_tails(settings: Any, bundle: Any) -> tuple[dict, Any]:
    """Tail models from the training rows of the bundle's own split."""
    from rail_risk.evaluation import (
        complete_days,
        load_snapshots,
        load_split_config,
        make_split,
    )

    cfg = load_split_config(Path("config/split.yaml"))
    snaps = load_snapshots(settings.data_dir / "features" / "snapshots")
    if cfg.routes:
        snaps = snaps[snaps["route_id"].isin(cfg.routes)]
    days = complete_days(
        settings.data_dir / "interim" / "hsp_manifest.sqlite",
        sorted(snaps["route_id"].unique()),
        cfg.windows_per_day,
    )
    split = make_split([d for d in days if d in set(snaps["service_date"])], cfg)
    train = snaps[snaps["service_date"].isin(set(split.train)) & snaps["label_delay_min"].notna()]
    return fit_route_tails(bundle.quantile, train)


def load_connection_inputs(settings: Any, bundle: Any) -> dict[str, Any]:
    log = get_logger("api")
    out: dict[str, Any] = {}
    try:
        out["stations"] = load_stations(Path("config/stations.yaml"))
        out["alert_tau"] = load_alert_config(Path("config/alerts.yaml")).tau
    except Exception as exc:  # noqa: BLE001 - /connection then answers 503
        log.error("connection_config_failed", error=repr(exc))
    try:
        from rail_risk import pmiss

        path = Path("config/pmiss_calibration.json")
        if path.exists():
            out["pmiss_calibration"] = pmiss.load(path)
    except Exception as exc:  # noqa: BLE001 - raw P(miss) is served instead
        log.error("pmiss_calibration_failed", error=repr(exc))
    try:
        out["outbound_history"] = outbound_history(settings.data_dir / "interim")
    except Exception as exc:  # noqa: BLE001
        log.error("outbound_history_failed", error=repr(exc))
    if bundle is not None:
        try:
            out["tails"], out["pooled_tail"] = route_tails(settings, bundle)
        except Exception as exc:  # noqa: BLE001
            log.error("tail_model_failed", error=repr(exc))
    return out
