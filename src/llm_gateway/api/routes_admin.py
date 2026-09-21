"""Admin endpoints — /v1/health (with breaker truth), /v1/cache/invalidate,
/v1/metrics, and the chaos tooling (mock targets only, dev/test only)."""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, model_validator

from llm_gateway import __version__
from llm_gateway.config import Environment
from llm_gateway.providers.base import ProviderRegistry
from llm_gateway.providers.mock_adapter import MockProvider
from llm_gateway.schemas.errors import CacheDisabled, ChaosDisabled, ChaosTargetError

router = APIRouter()


class ProviderHealth(BaseModel):
    name: str
    healthy: bool
    # Request-outcome truth, not endpoint truth: the live Day-5 finding was
    # /v1/models healthy while generateContent 503'd. Breaker state derives
    # from actual request conclusions.
    breaker: str


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
    def _exactly_one(self) -> "InvalidateRequest":
        provided = sum([bool(self.prompt_hash), bool(self.model_family), self.flush])
        if provided != 1:
            raise ValueError("provide exactly one of: prompt_hash | model_family | flush=true")
        return self


def _chaos_guard(request: Request) -> None:
    env = request.app.state.settings.app.environment
    if env not in (Environment.DEVELOPMENT, Environment.TEST):
        raise ChaosDisabled()


def _chaos_target(name: str, request: Request) -> MockProvider:
    registry: ProviderRegistry = request.app.state.providers
    try:
        provider = registry.get(name)
    except LookupError:
        raise ChaosTargetError(f"no provider named {name!r}") from None
    if not isinstance(provider, MockProvider):
        raise ChaosTargetError(
            f"chaos targets must be mock providers; {name!r} is a {type(provider).__name__} — "
            "remote providers cannot be killed by honest tooling"
        )
    return provider


@router.get("/v1/health", response_model=HealthResponse, summary="Gateway + per-provider health")
async def health(request: Request) -> HealthResponse:
    app = request.app
    registry: ProviderRegistry = app.state.providers
    breakers = getattr(app.state, "breakers", None)
    snapshot = breakers.snapshot() if breakers is not None else {}
    adapters = registry.all()
    results = await asyncio.gather(*(a.health_check() for a in adapters))
    providers = [
        ProviderHealth(name=a.name, healthy=ok, breaker=snapshot.get(a.name, "closed"))
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


@router.get("/v1/metrics", summary="Prometheus scrape endpoint")
async def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/v1/chaos/providers/{name}/kill", summary="Chaos: force a mock provider unhealthy")
async def chaos_kill(name: str, request: Request) -> dict:
    _chaos_guard(request)
    provider = _chaos_target(name, request)
    provider.kill()
    return {"provider": name, "killed": True}


@router.post("/v1/chaos/providers/{name}/revive", summary="Chaos: revive a killed mock provider")
async def chaos_revive(name: str, request: Request) -> dict:
    _chaos_guard(request)
    provider = _chaos_target(name, request)
    provider.revive()
    return {"provider": name, "revived": True}
