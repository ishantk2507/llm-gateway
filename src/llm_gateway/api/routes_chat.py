"""POST /v1/chat/completions — the Day-1 request lifecycle (DESIGN.md §6).

Today's path: validate → stream check → dropped-param warning → provider →
headers + structured log. Each later phase inserts exactly one step: auth and
rate limiting (Day 6) before the provider lookup; cache (Day 3) and router
(Day 4) around ``_resolve_provider``. The skeleton's job is to make those
insertions boring.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, Response

from llm_gateway.config import Settings
from llm_gateway.observability.logging import bind_log_fields, get_logger
from llm_gateway.providers.base import ProviderAdapter, ProviderRegistry
from llm_gateway.schemas.errors import RequestTimedOut, StreamNotSupported
from llm_gateway.schemas.openai_api import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()
logger = get_logger(__name__)


def _resolve_provider(model: str, registry: ProviderRegistry) -> ProviderAdapter:
    """Day-1 stand-in for the router.

    Day 4 replaces this function — and only this function — with the
    rule-based classifier: 'mock' pins the mock, everything else gets the
    first healthy provider in the default chain.
    """
    if model == "mock":
        return registry.get("mock")
    return registry.default_chain()[0]


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create a chat completion (OpenAI-compatible)",
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    fastapi_response: Response,
    http_request: Request,
) -> ChatCompletionResponse:
    # 1 — stream is a documented 400, never a crash, never a silent downgrade.
    if request.stream:
        raise StreamNotSupported()

    # 2 — unknown parameters are dropped, loudly (§5.1: parameter whitelist).
    dropped = request.dropped_parameters()
    if dropped:
        logger.warning(
            "dropped_parameters", dropped=dropped, note="not supported in v1; not forwarded"
        )

    settings: Settings = http_request.app.state.settings
    registry: ProviderRegistry = http_request.app.state.providers
    adapter = _resolve_provider(request.model, registry)

    # 3 — call the provider inside the per-attempt timeout budget (§9).
    bind_log_fields(model=request.model, provider_used=adapter.name)
    try:
        completion = await asyncio.wait_for(
            adapter.complete(request),
            timeout=settings.reliability.per_attempt_timeout_s,
        )
    except TimeoutError as exc:
        raise RequestTimedOut(
            f"provider {adapter.name!r} exceeded the per-attempt timeout "
            f"({settings.reliability.per_attempt_timeout_s}s)"
        ) from exc

    # 4 — cost. Day 5's pricing.py swaps in behind this exact line.
    cost_usd = 0.0

    # 5 — the request log line gets its final fields (§5.5).
    usage = completion.usage
    bind_log_fields(
        routing_tier="auto",  # honest placeholder — the router lands Day 4
        cache_hit=False,  # the cache lands Day 3
        tokens_in=usage.prompt_tokens,
        tokens_out=usage.completion_tokens,
        cost_usd=cost_usd,
    )

    # 6 — the four contract headers, always present (§7.2).
    fastapi_response.headers["x-cache-hit"] = "false"
    fastapi_response.headers["x-provider-used"] = adapter.name
    fastapi_response.headers["x-routing-tier"] = "auto"
    fastapi_response.headers["x-cost-usd"] = f"{cost_usd:.4f}"
    return completion
