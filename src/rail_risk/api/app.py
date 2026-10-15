"""FastAPI app: live single-leg risk with explicit fallbacks."""

import asyncio
import contextlib
import os
import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from rail_risk.api.schemas import Health, RiskRequest, RiskResponse
from rail_risk.api.service import NotFound, RiskService
from rail_risk.logging import get_logger

REQUEST_BUDGET_SECONDS = 2.0
REFRESH_SECONDS = 30.0
RID_RE = re.compile(r"^\d{15}$")


def create_app(
    service_factory: Callable[[], RiskService],
    *,
    alias: str = "champion",
    refresh_seconds: float = REFRESH_SECONDS,
    budget_seconds: float = REQUEST_BUDGET_SECONDS,
) -> FastAPI:
    """`service_factory` builds the RiskService at startup, so the model, Darwin state and
    feature service are loaded once. Tests pass a factory over fixture data."""
    log = get_logger("api")
    state: dict[str, Any] = {"service": None, "task": None}

    async def refresher(service: RiskService) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                await loop.run_in_executor(None, service.store.refresh)
                await loop.run_in_executor(None, service.refresh_features)
            except Exception as exc:  # noqa: BLE001 - a bad refresh must not stop serving
                log.error("refresh_failed", error=repr(exc))
            await asyncio.sleep(refresh_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        state["service"] = await loop.run_in_executor(None, service_factory)
        if refresh_seconds > 0:
            state["task"] = asyncio.create_task(refresher(state["service"]))
        yield
        task = state["task"]
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Connection Risk Engine", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", request_id):
            request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.monotonic()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round((time.monotonic() - started) * 1000, 1),
        )
        return response

    @app.exception_handler(ValidationError)
    async def validation_error(request: Request, exc: ValidationError):
        return JSONResponse(status_code=422, content={"detail": "invalid request"})

    @app.get("/health", response_model=Health)
    async def health() -> Health:
        service: RiskService = state["service"]
        age = service.age_minutes()
        stale = service.is_stale()
        fallback = service.bundle is None
        return Health(
            status="degraded" if (stale or fallback) else "ok",
            model_alias=alias,
            model_run_id=None if fallback else service.model_run_id,
            model_fallback=fallback,
            darwin_data_age_minutes=None if age is None else round(age, 1),
            stale_live_data=stale,
            watched_services=len(service.active_rids()),
        )

    @app.post("/risk", response_model=RiskResponse)
    async def risk(body: RiskRequest, request: Request) -> RiskResponse:
        service: RiskService = state["service"]
        loop = asyncio.get_running_loop()
        call = lambda: service.risk(  # noqa: E731
            body.rid, body.fare, body.ticket, body.route, request.state.request_id
        )
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, call), timeout=budget_seconds)
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except TimeoutError:
            log.error("request_timeout", request_id=request.state.request_id, rid=body.rid)
            raise HTTPException(
                status_code=504, detail=f"no answer within {budget_seconds:.0f} seconds"
            ) from None

    return app


def default_service() -> RiskService:
    """Production wiring: champion model by alias, Darwin state from data/, Spark features."""
    from rail_risk.config import Settings, load_routes
    from rail_risk.evaluation import load_bands
    from rail_risk.live import FeatureService, LiveStore, load_tiplocs
    from rail_risk.models import registry
    from rail_risk.repay import load_rules
    from rail_risk.spark import build_spark

    log = get_logger("api")
    settings = Settings()
    data = settings.data_dir
    bands = load_bands(Path("config/delay_bands.yaml"))
    store = LiveStore(data / "interim", data / "raw" / "darwin", load_tiplocs())
    store.refresh()
    spark = build_spark("risk-api", cores=2, driver_memory="2g", shuffle_partitions=2,
                        adaptive=False)  # fmt: skip
    features = FeatureService(spark, data / "interim", settings.routes_file)
    bundle = baseline = run_id = version = None
    trained_on: set[str] = set()
    try:
        loaded = registry.load_alias(Path(os.environ.get("RAIL_MLFLOW_DIR", "mlruns")))
        bundle, baseline = loaded.bundle, loaded.baseline
        run_id, version = loaded.run_id, loaded.version
        trained_on = set(bundle.quantile.chosen.design.categories.get("route_id", []))
    except Exception as exc:  # noqa: BLE001 - any load failure falls back to the baseline
        log.error("model_load_failed", error=repr(exc))
    service = RiskService(
        store=store, features=features, routes=load_routes(settings.routes_file).routes,
        bands=bands, rules=load_rules(Path("config/delay_repay.yaml")), bundle=bundle,
        baseline=baseline, model_run_id=run_id, model_version=version,
        route_model_trained_on=trained_on,
    )  # fmt: skip
    if bundle is None:
        service.baseline = baseline or _fit_baseline(settings, bands)
    service.refresh_features()
    return service


def _fit_baseline(settings: Any, bands: Any) -> Any:
    """Fit the historical baseline from the training snapshots when no bundle could be loaded."""
    from rail_risk.evaluation import (
        HistoricalBaseline,
        complete_days,
        load_snapshots,
        load_split_config,
        make_split,
    )

    try:
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
        train = snaps[
            snaps["service_date"].isin(set(split.train)) & snaps["label_delay_min"].notna()
        ]
        return HistoricalBaseline(bands).fit(train)
    except Exception as exc:  # noqa: BLE001
        get_logger("api").error("baseline_fit_failed", error=repr(exc))
        return None


def app_from_environment() -> FastAPI:
    return create_app(default_service)
