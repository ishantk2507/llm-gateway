"""POST /v1/chat/completions — request lifecycle (DESIGN.md §6), Day-3 edition.

validate → stream check → dropped-param warning → CACHE (exact → semantic) →
chain resolution → budgeted fallback walk → write-through → headers + log.
Day 4 replaces _resolve_chain with the router. Nothing else moves.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from llm_gateway.config import Settings
from llm_gateway.observability.logging import bind_log_fields, get_logger
from llm_gateway.providers.base import ProviderAdapter, ProviderRegistry
from llm_gateway.reliability.fallback import execute_with_fallback
from llm_gateway.reliability.retry import TimeoutBudget
from llm_gateway.schemas.errors import StreamNotSupported
from llm_gateway.schemas.openai_api import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()
logger = get_logger(__name__)


def _resolve_chain(model: str, registry: ProviderRegistry) -> list[ProviderAdapter]:
    """Day-2 stand-in for the router (Day 4 replaces this function wholesale)."""
    if model == "mock":
        return [registry.get("mock")]
    if model.startswith(("gpt-", "o1", "o3", "chatgpt")):
        if "openai" in registry.names():
            return [registry.get("openai")]
        logger.warning("pinned_provider_not_configured", model=model, action="using_default_chain")
    if model.startswith("claude-"):
        if "anthropic" in registry.names():
            return [registry.get("anthropic")]
        logger.warning("pinned_provider_not_configured", model=model, action="using_default_chain")
    return registry.default_chain()


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
    if request.stream:
        raise StreamNotSupported()

    dropped = request.dropped_parameters()
    if dropped:
        logger.warning(
            "dropped_parameters", dropped=dropped, note="not supported in v1; not forwarded"
        )

    settings: Settings = http_request.app.state.settings
    registry: ProviderRegistry = http_request.app.state.providers
    cache = getattr(http_request.app.state, "cache", None)
    bypass = http_request.headers.get("x-cache", "").lower() == "bypass"

    # ── cache lookup: hits never touch a provider, a budget, or a bill ──
    cache_result = None
    if cache is not None and not bypass:
        cache_result = await cache.lookup(request)
        if cache_result.hit:
            usage = cache_result.response.usage
            fields: dict = {
                "model": request.model,
                "provider_used": "cache",
                "routing_tier": "cache",
                "cache_hit": True,
                "cache_layer": cache_result.layer,
                "tokens_in": usage.prompt_tokens,
                "tokens_out": usage.completion_tokens,
                "cost_usd": 0.0,
            }
            if cache_result.similarity is not None:
                fields["similarity_score"] = cache_result.similarity
            bind_log_fields(**fields)
            fastapi_response.headers["x-cache-hit"] = "true"
            fastapi_response.headers["x-provider-used"] = "cache"
            fastapi_response.headers["x-routing-tier"] = "cache"
            fastapi_response.headers["x-cost-usd"] = "0.0000"
            return cache_result.response

    # ── miss: the full Day-2 provider path, unchanged ──
    chain = _resolve_chain(request.model, registry)
    budget = TimeoutBudget(settings.reliability.request_timeout_budget_s)
    bind_log_fields(model=request.model, provider_chain=[p.name for p in chain])
    if cache_result is not None and cache_result.near_miss:
        # Calibration data (§5.2): close-but-not-cached, fed to Day 6's sweep
        bind_log_fields(near_miss=True, similarity_score=cache_result.similarity)

    completion, provider_name = await execute_with_fallback(
        chain, request, budget=budget, settings=settings.reliability
    )

    if cache is not None:  # write-through (ADR-0002) — after success only
        await cache.store(request, completion)

    cost_usd = 0.0  # Day 5's pricing.py swaps in behind this exact line
    usage = completion.usage
    bind_log_fields(
        routing_tier="auto",
        cache_hit=False,
        provider_used=provider_name,
        tokens_in=usage.prompt_tokens,
        tokens_out=usage.completion_tokens,
        cost_usd=cost_usd,
    )
    fastapi_response.headers["x-cache-hit"] = "false"
    fastapi_response.headers["x-provider-used"] = provider_name
    fastapi_response.headers["x-routing-tier"] = "auto"
    fastapi_response.headers["x-cost-usd"] = f"{cost_usd:.4f}"
    return completion
