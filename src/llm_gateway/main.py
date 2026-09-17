"""App factory + lifespan. Tests inject providers= and cache=; production
builds from settings and fails loudly when infrastructure is missing."""

from __future__ import annotations

import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_gateway import __version__
from llm_gateway.api.routes_admin import router as admin_router
from llm_gateway.api.routes_chat import router as chat_router
from llm_gateway.cache.service import CacheService, build_cache
from llm_gateway.config import Environment, Settings, get_settings
from llm_gateway.observability.logging import (
    RequestLoggingMiddleware,
    configure_logging,
    get_logger,
)
from llm_gateway.providers.base import ProviderRegistry, build_registry
from llm_gateway.schemas.errors import install_exception_handlers

logger = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    providers: ProviderRegistry | None = None,
    cache: CacheService | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    configure_logging(
        settings.app.log_level,
        pretty=settings.app.environment == Environment.DEVELOPMENT and sys.stderr.isatty(),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.providers = providers or build_registry(settings)
        app.state.started_at = time.monotonic()

        from llm_gateway.router.mapping import build_router

        app.state.router = (
            build_router(
                app.state.providers, settings.router.keywords_path, settings.router.tiers_path
            )
            if settings.router.enabled
            else None
        )

        # Test apps inject their own cache (fakeredis + fake embedder); auto-
        # building here would demand a live Redis in CI — that's a test
        # concern, not a runtime one.
        redis_client = None
        app.state.cache = cache
        if (
            cache is None
            and settings.cache.enabled
            and settings.app.environment != Environment.TEST
        ):
            service, redis_client = await build_cache(settings)
            app.state.cache = service

        logger.info(
            "gateway_started",
            version=__version__,
            environment=settings.app.environment.value,
            providers=[p.name for p in app.state.providers.all()],
            cache_enabled=app.state.cache is not None,
            router_enabled=app.state.router is not None,
        )
        yield

        if app.state.cache is not None:
            await app.state.cache.aclose()
        if redis_client is not None:
            await redis_client.aclose()
        await app.state.providers.close_all()

    app = FastAPI(
        title=settings.app.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(RequestLoggingMiddleware)
    install_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(admin_router)
    return app


app = create_app()
