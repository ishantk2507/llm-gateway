"""Admin endpoints — /v1/health today; /v1/cache/invalidate joins on Day 3."""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from llm_gateway import __version__
from llm_gateway.providers.base import ProviderRegistry

router = APIRouter()


class ProviderHealth(BaseModel):
    name: str
    healthy: bool


class HealthResponse(BaseModel):
    # "ok" = at least one provider healthy — that's the real availability
    # question. Day 5 adds circuit_state per provider without breaking readers.
    status: Literal["ok", "degraded"]
    version: str
    uptime_s: float
    environment: str
    providers: list[ProviderHealth]


@router.get("/v1/health", response_model=HealthResponse, summary="Gateway + per-provider health")
async def health(request: Request) -> HealthResponse:
    app = request.app
    registry: ProviderRegistry = app.state.providers
    adapters = registry.all()

    results = await asyncio.gather(*(adapter.health_check() for adapter in adapters))
    providers = [
        ProviderHealth(name=adapter.name, healthy=ok)
        for adapter, ok in zip(adapters, results, strict=True)
    ]
    any_healthy = any(p.healthy for p in providers)

    return HealthResponse(
        status="ok" if any_healthy else "degraded",
        version=__version__,
        uptime_s=round(time.monotonic() - app.state.started_at, 1),
        environment=app.state.settings.app.environment.value,
        providers=providers,
    )
