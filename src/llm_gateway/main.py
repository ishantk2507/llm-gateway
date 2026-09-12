"""App factory + lifespan.

``create_app(settings)`` exists so tests build isolated apps with their own
registries; the module-level ``app`` is what uvicorn serves (``make run``).
"""

from __future__ import annotations

import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_gateway import __version__
from llm_gateway.api.routes_admin import router as admin_router
from llm_gateway.api.routes_chat import router as chat_router
from llm_gateway.config import Environment, Settings, get_settings
from llm_gateway.observability.logging import (
    RequestLoggingMiddleware,
    configure_logging,
    get_logger,
)
from llm_gateway.providers.base import build_registry
from llm_gateway.schemas.errors import install_exception_handlers

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    configure_logging(
        settings.app.log_level,
        pretty=settings.app.environment == Environment.DEVELOPMENT and sys.stderr.isatty(),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: assemble the world. Day 3 adds the redis pool + FAISS
        # index here; shutdown will flush and close them.
        app.state.settings = settings
        app.state.providers = build_registry(settings)
        app.state.started_at = time.monotonic()
        logger.info(
            "gateway_started",
            version=__version__,
            environment=settings.app.environment.value,
            providers=[p.name for p in app.state.providers.all()],
        )
        yield

    app = FastAPI(
        title=settings.app.app_name,
        version=__version__,
        lifespan=lifespan,  # OpenAPI docs at :8000/docs — the demo's first stop
    )
    app.add_middleware(RequestLoggingMiddleware)
    install_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(admin_router)
    return app


app = create_app()
