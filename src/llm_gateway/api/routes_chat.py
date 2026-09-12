"""POST /v1/chat/completions — request lifecycle (DESIGN.md §6), Day-2 edition.

validate → stream check → dropped-param warning → chain resolution → budgeted
fallback walk → headers + structured log. Day 3 inserts the cache between the
stream check and the walk; Day 4 replaces _resolve_chain with the router.
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
    """Day-2 stand-in for the router (Day 4 replaces this function wholesale).

    'mock' pins the mock; a real model name pins its provider by prefix — an
    unconfigured pin falls through to the chain, loudly; anything virtual
    ('auto', tier names) walks the full default chain.
    """
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
    chain = _resolve_chain(request.model, registry)

    # One clock per request, shared by retry + fallback (§9).
    budget = TimeoutBudget(settings.reliability.request_timeout_budget_s)

    bind_log_fields(model=request.model, provider_chain=[p.name for p in chain])
    completion, provider_name = await execute_with_fallback(
        chain, request, budget=budget, settings=settings.reliability
    )

    cost_usd = 0.0  # Day 5's pricing.py swaps in behind this exact line
    usage = completion.usage
    bind_log_fields(
        routing_tier="auto",  # honest placeholder — the router lands Day 4
        cache_hit=False,  # the cache lands Day 3
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
