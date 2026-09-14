"""Admin endpoints — /v1/health + /v1/cache/invalidate (DESIGN.md §7)."""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, model_validator

from llm_gateway import __version__
from llm_gateway.providers.base import ProviderRegistry
from llm_gateway.schemas.errors import CacheDisabled

router = APIRouter()


class ProviderHealth(BaseModel):
    name: str
    healthy: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    uptime_s: float
    environment: str
    providers: list[ProviderHealth]


class InvalidateRequest(BaseModel):
    prompt_hash: str | None = None
    model_family: str | None = None
    flush: bool = False

    @model_validator(mode="after")
    def _exactly_one(self) -> InvalidateRequest:
        provided = sum([bool(self.prompt_hash), bool(self.model_family), self.flush])
        if provided != 1:
            raise ValueError("provide exactly one of: prompt_hash | model_family | flush=true")
        return self


@router.get("/v1/health", response_model=HealthResponse, summary="Gateway + per-provider health")
async def health(request: Request) -> HealthResponse:
    app = request.app
    registry: ProviderRegistry = app.state.providers
    adapters = registry.all()
    results = await asyncio.gather(*(a.health_check() for a in adapters))
    providers = [
        ProviderHealth(name=a.name, healthy=ok)
        for a, ok in zip(adapters, results, strict=True)
    ]
    return HealthResponse(
        status="ok" if any(p.healthy for p in providers) else "degraded",
        version=__version__,
        uptime_s=round(time.monotonic() - app.state.started_at, 1),
        environment=app.state.settings.app.environment.value,
        providers=providers,
    )


@router.post("/v1/cache/invalidate", summary="Evict cache entries")
async def invalidate_cache(body: InvalidateRequest, request: Request) -> dict:
    cache = getattr(request.app.state, "cache", None)
    if cache is None:
        raise CacheDisabled()
    evicted = await cache.invalidate(
        prompt_hash=body.prompt_hash, model_family=body.model_family, flush=body.flush
    )
    return {"evicted": evicted}