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

from rail_risk.api.schemas import (
    ConnectionRequest,
    ConnectionResponse,
    Health,
    RiskRequest,
    RiskResponse,
    RouteItem,
    ServiceItem,
    Warming,
)
from rail_risk.api.service import ConnectionUnavailable, NotFound, RiskService
from rail_risk.api.service import Warming as WarmingError
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
    state: dict[str, Any] = {"service": None, "task": None, "wake": None}

    async def refresher(service: RiskService) -> None:
        """Rebuild state every `refresh_seconds`, or at once when a request missed the cache."""
        loop = asyncio.get_running_loop()
        first = True
        while True:
            try:
                if not first:
                    await loop.run_in_executor(None, service.store.refresh)
                try:  # a parse failure must not stop the feature refresh
                    await loop.run_in_executor(None, service.parse_closed_hours)
                except Exception as exc:  # noqa: BLE001
                    log.error("parse_failed", error=repr(exc))
                await loop.run_in_executor(None, service.refresh_features)
                first = False
            except Exception as exc:  # noqa: BLE001 - a bad refresh must not stop serving
                log.error("refresh_failed", error=repr(exc))
            state["wake"].clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(state["wake"].wait(), timeout=refresh_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        state["service"] = await loop.run_in_executor(None, service_factory)
        state["wake"] = asyncio.Event()
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
            status="degraded" if (stale or fallback or service.parse_is_stale()) else "ok",
            model_alias=alias,
            model_run_id=None if fallback else service.model_run_id,
            model_fallback=fallback,
            darwin_data_age_minutes=None if age is None else round(age, 1),
            stale_live_data=stale,
            watched_services=len(service.active_rids()),
            stale_parse=service.parse_is_stale(),
            parse_lag_hours=(
                None
                if service.parse_state is None
                else None
                if service.parse_state.lag_hours is None
                else round(service.parse_state.lag_hours, 1)
            ),
        )

    @app.get("/routes", response_model=list[RouteItem])
    async def routes() -> list[RouteItem]:
        service: RiskService = state["service"]
        return [
            RouteItem(
                id=r.id, origin=r.origin, destination=r.destination,
                model_trained_on_route=r.id in service.route_model_trained_on,
            )
            for r in service.routes
        ]  # fmt: skip

    @app.get("/services", response_model=list[ServiceItem])
    async def services(route: str = "bhm-eus") -> list[ServiceItem]:
        """Services due at the route destination in the next three hours, soonest first."""
        service: RiskService = state["service"]
        return [ServiceItem(**s) for s in service.services(route)]

    @app.get("/inbound", response_model=list[ServiceItem])
    async def inbound() -> list[ServiceItem]:
        """Aston to New Street services due at New Street in the next two hours."""
        service: RiskService = state["service"]
        return [ServiceItem(**s) for s in service.inbound_services(120)]

    @app.post(
        "/connection",
        response_model=ConnectionResponse,
        responses={202: {"model": Warming}},
    )
    async def connection(
        body: ConnectionRequest, request: Request
    ) -> ConnectionResponse | JSONResponse:
        service: RiskService = state["service"]
        loop = asyncio.get_running_loop()
        call = lambda: service.connection(  # noqa: E731
            body.inbound_rid,
            body.outbound_departure,
            body.fare,
            body.ticket,
            request.state.request_id,
        )
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, call), timeout=budget_seconds)
        except WarmingError as exc:
            if state["wake"] is not None:
                state["wake"].set()
            warm = Warming(
                request_id=request.state.request_id, rid=exc.rid,
                retry_after_seconds=exc.retry_after,
                detail="features for the inbound service are being built; retry shortly",
            )  # fmt: skip
            return JSONResponse(
                status_code=202,
                content=warm.model_dump(),
                headers={"Retry-After": str(exc.retry_after)},
            )
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except ConnectionUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except TimeoutError:
            log.error("request_timeout", request_id=request.state.request_id, rid=body.inbound_rid)
            raise HTTPException(
                status_code=504, detail=f"no answer within {budget_seconds:.0f} seconds"
            ) from None

    @app.post("/risk", response_model=RiskResponse, responses={202: {"model": Warming}})
    async def risk(body: RiskRequest, request: Request) -> RiskResponse | JSONResponse:
        service: RiskService = state["service"]
        loop = asyncio.get_running_loop()
        call = lambda: service.risk(  # noqa: E731
            body.rid, body.fare, body.ticket, body.route, request.state.request_id
        )
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, call), timeout=budget_seconds)
        except WarmingError as exc:
            if state["wake"] is not None:
                state["wake"].set()  # build this service now rather than at the next tick
            body = Warming(
                request_id=request.state.request_id, rid=exc.rid,
                retry_after_seconds=exc.retry_after,
                detail="features for this service are being built; retry shortly",
            )  # fmt: skip
            return JSONResponse(
                status_code=202,
                content=body.model_dump(),
                headers={"Retry-After": str(exc.retry_after)},
            )
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except TimeoutError:
            log.error("request_timeout", request_id=request.state.request_id, rid=body.rid)
            raise HTTPException(
                status_code=504, detail=f"no answer within {budget_seconds:.0f} seconds"
            ) from None

    return app


def default_service() -> RiskService:
    """Production wiring: champion model by alias, Darwin state from data/, Spark features.

    The slow parts are independent, so they run concurrently: loading the Darwin state, starting
    Spark with the history labels, and loading the model. The first feature batch needs the first
    two only, so it starts as soon as they finish, while the model may still be loading.
    """
    from concurrent.futures import ThreadPoolExecutor

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

    def load_state() -> LiveStore:
        store.refresh()
        return store

    def start_features() -> FeatureService:
        spark = build_spark(
            "risk-api", cores=2, driver_memory="2g", shuffle_partitions=2, adaptive=False
        )
        return FeatureService(spark, data / "interim", settings.routes_file)

    def load_model():
        try:
            loaded = registry.load_alias(Path(os.environ.get("RAIL_MLFLOW_DIR", "mlruns")))
        except Exception as exc:  # noqa: BLE001 - any load failure falls back to the baseline
            log.error("model_load_failed", error=repr(exc))
            return None
        return loaded

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_state, f_feat, f_model = (
            pool.submit(load_state), pool.submit(start_features), pool.submit(load_model)
        )  # fmt: skip
        f_state.result()
        features = f_feat.result()
        loaded = f_model.result()
    bundle = baseline = run_id = version = None
    trained_on: set[str] = set()
    if loaded is not None:
        bundle, baseline = loaded.bundle, loaded.baseline
        run_id, version = loaded.run_id, loaded.version
        trained_on = set(bundle.quantile.chosen.design.categories.get("route_id", []))
    service = RiskService(
        store=store, features=features, routes=load_routes(settings.routes_file).routes,
        bands=bands, rules=load_rules(Path("config/delay_repay.yaml")), bundle=bundle,
        baseline=baseline, model_run_id=run_id, model_version=version,
        route_model_trained_on=trained_on,
    )  # fmt: skip
    if bundle is None:
        service.baseline = baseline or _fit_baseline(settings, bands)
    from rail_risk.api.connection_inputs import load_connection_inputs

    for name, value in load_connection_inputs(settings, bundle).items():
        setattr(service, name, value)
    service.parse_enabled = os.access(data / "interim", os.W_OK) and (
        os.environ.get("RAIL_PARSE_IN_API", "1") != "0"
    )
    try:
        service.parse_closed_hours()  # catches up on hours closed while the API was down
    except Exception as exc:  # noqa: BLE001
        log.error("parse_failed", error=repr(exc))
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
