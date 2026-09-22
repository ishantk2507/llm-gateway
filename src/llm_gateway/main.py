"""App factory + lifespan. Tests inject providers=/cache=/pricing=/
repository=/breakers=/rate_limiter=; production builds from settings and
fails loudly when infrastructure is missing."""

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
from llm_gateway.observability.metrics import set_breaker_state
from llm_gateway.observability.pricing import PricingService
from llm_gateway.observability.repository import RequestRepository
from llm_gateway.providers.base import ProviderRegistry, build_registry
from llm_gateway.reliability.circuit_breaker import BreakerBoard
from llm_gateway.reliability.rate_limiter import TokenBucketLimiter
from llm_gateway.schemas.errors import install_exception_handlers

logger = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    providers: ProviderRegistry | None = None,
    cache: CacheService | None = None,
    pricing: PricingService | None = None,
    repository: RequestRepository | None = None,
    breakers: BreakerBoard | None = None,
    rate_limiter: TokenBucketLimiter | None = None,
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

        # Test apps inject their own cache/pricing/repository (fakeredis, tmp
        # paths, round-number tables); auto-building in TEST would demand a
        # live Redis and write to the repo — test concerns, not runtime ones.
        redis_client = None
        app.state.cache = cache
        if (
            cache is None
            and settings.cache.enabled
            and settings.app.environment != Environment.TEST
        ):
            service, redis_client = await build_cache(settings)
            app.state.cache = service

        app.state.pricing = pricing
        app.state.repository = repository
        if settings.app.environment != Environment.TEST:
            if pricing is None:
                app.state.pricing = PricingService(settings.observability.pricing_path)
            if repository is None:
                app.state.repository = RequestRepository(settings.observability.database_url)

        # Breakers: pure in-memory — always on, even in TEST (tests inject
        # their own board with a fake clock when they need determinism).
        app.state.breakers = breakers or BreakerBoard(
            settings.reliability, on_state=set_breaker_state
        )
        # Light the gauge for every registered provider: absence reads as
        # "no data", which is a lie on day one of the panel's life.
        for provider in app.state.providers.all():
            set_breaker_state(provider.name, 0)

        # Rate limiter: only when keys are configured. Reuses the cache's
        # Redis pool when present; otherwise a lazy client of its own —
        # redis-py connects on first command, so zero-infra boot is preserved
        # (first request logs rate_limiter_unavailable and fails open).
        lazy_redis = None
        app.state.rate_limiter = rate_limiter
        if rate_limiter is None and settings.auth.api_keys:
            if redis_client is not None:
                client = redis_client
            else:
                from redis import asyncio as aioredis

                client = aioredis.from_url(settings.redis.url)
                lazy_redis = client
            app.state.rate_limiter = TokenBucketLimiter(
                client, rpm=settings.auth.rate_limit_rpm, burst=settings.auth.rate_limit_burst
            )

        logger.info(
            "gateway_started",
            version=__version__,
            environment=settings.app.environment.value,
            providers=[p.name for p in app.state.providers.all()],
            cache_enabled=app.state.cache is not None,
            router_enabled=app.state.router is not None,
            observability_enabled=app.state.repository is not None,
            auth_enabled=bool(settings.auth.api_keys),
        )
        yield

        if app.state.cache is not None:
            await app.state.cache.aclose()
        if redis_client is not None:
            await redis_client.aclose()
        if lazy_redis is not None:
            await lazy_redis.aclose()
        if app.state.repository is not None:
            await app.state.repository.aclose()
        await app.state.providers.close_all()

    app = FastAPI(title=settings.app.app_name, version=__version__, lifespan=lifespan)
    app.add_middleware(RequestLoggingMiddleware)
    install_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(admin_router)
    return app
