"""POST /v1/chat/completions — request lifecycle (DESIGN.md §6), Day-5 edition.

validate → stream check → dropped-param warning → CACHE (exact → semantic) →
ROUTE (header pin > model pin > classifier) → budgeted fallback walk →
write-through → COST (pricing table, provider-reported usage) → headers.

The middleware handles metrics + persistence from the bound fields; this
route's only new job is computing cost_usd / cost_saved_usd and binding them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from llm_gateway.api.dependencies import gateway_gate
from llm_gateway.config import Settings, Tier
from llm_gateway.observability.logging import bind_log_fields, get_logger
from llm_gateway.providers.base import ProviderAdapter, ProviderRegistry
from llm_gateway.reliability.circuit_breaker import BreakerBoard
from llm_gateway.reliability.fallback import execute_with_fallback
from llm_gateway.reliability.retry import TimeoutBudget
from llm_gateway.router.mapping import RouterService
from llm_gateway.schemas.errors import StreamNotSupported
from llm_gateway.schemas.openai_api import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()
logger = get_logger(__name__)

_HEADER_TIERS = {"cheap": Tier.CHEAP, "standard": Tier.STANDARD, "premium": Tier.PREMIUM}


def _resolve_route(
    request: ChatCompletionRequest, http_request: Request, router_svc: RouterService | None
) -> tuple[list[ProviderAdapter], str, str]:
    """→ (chain, routing_tier_label, rule_fired), per DESIGN.md §5.3."""
    header_tier = http_request.headers.get("x-routing-strategy", "").lower().strip()

    if router_svc is not None and header_tier in _HEADER_TIERS:
        tier = _HEADER_TIERS[header_tier]  # the client knows better — let them
        return router_svc.chain_for(tier), tier.value, "header_override"

    if request.model == "mock":
        registry: ProviderRegistry = http_request.app.state.providers
        return [registry.get("mock")], "pinned", "model_pin"

    if router_svc is not None and (pinned := router_svc.pinned(request.model)) is not None:
        return [pinned], "pinned", "model_pin"

    if router_svc is not None:
        target = router_svc.pin_target(request.model)
        if target is not None:  # pin names a provider that isn't registered
            logger.warning(
                "pinned_provider_not_configured",
                model=request.model,
                target=target,
                action="routing_instead",
            )
        decision = router_svc.classify(request)
        return router_svc.chain_for(decision.tier), decision.tier.value, decision.rule_fired

    return http_request.app.state.providers.default_chain(), "auto", "router_disabled"


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(gateway_gate)],  # auth + rate limit — chat route ONLY
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
    # registry: ProviderRegistry = http_request.app.state.providers
    router_svc: RouterService | None = getattr(http_request.app.state, "router", None)
    if router_svc is None or not settings.router.enabled:
        router_svc = None
    cache = getattr(http_request.app.state, "cache", None)
    pricing = getattr(http_request.app.state, "pricing", None)
    bypass = http_request.headers.get("x-cache", "").lower() == "bypass"

    # ── cache lookup: hits never touch a provider, a router, or a bill ──
    cache_result = None
    if cache is not None and not bypass:
        cache_result = await cache.lookup(request)
        if cache_result.hit:
            # A hit costs nothing — and SAVES what the stored response would
            # have cost: the cached entry carries the model + usage it was
            # served with, so price exactly what you didn't spend.
            saved_usd = 0.0
            if pricing is not None:
                saved_usd = pricing.estimate(
                    cache_result.response.model, cache_result.response.usage
                ).cost_usd
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
                "cost_saved_usd": saved_usd,
            }
            if cache_result.similarity is not None:
                fields["similarity_score"] = cache_result.similarity
            bind_log_fields(**fields)
            fastapi_response.headers["x-cache-hit"] = "true"
            fastapi_response.headers["x-provider-used"] = "cache"
            fastapi_response.headers["x-routing-tier"] = "cache"
            fastapi_response.headers["x-cost-usd"] = "0.0000"
            return cache_result.response

    # ── routing ──
    chain, tier_label, rule_fired = _resolve_route(request, http_request, router_svc)

    # ── miss: budgeted fallback walk ──
    budget = TimeoutBudget(settings.reliability.request_timeout_budget_s)
    bind_log_fields(
        model=request.model,
        provider_chain=[p.name for p in chain],
        routing_tier=tier_label,
        rule_fired=rule_fired,
    )
    if cache_result is not None and cache_result.near_miss:
        bind_log_fields(near_miss=True, similarity_score=cache_result.similarity)

    breakers: BreakerBoard | None = getattr(http_request.app.state, "breakers", None)
    completion, provider_name = await execute_with_fallback(
        chain, request, budget=budget, settings=settings.reliability, breakers=breakers
    )

    if cache is not None:  # write-through — after success only
        await cache.store(request, completion)

    # ── cost: published rates × provider-reported usage. Never estimated. ──
    cost_usd = 0.0
    if pricing is not None:
        estimate = pricing.estimate(completion.model, completion.usage)
        cost_usd = estimate.cost_usd
        if not estimate.priced:
            bind_log_fields(priced=False)  # never a silent zero (ADR-0008 rule)

    usage = completion.usage
    bind_log_fields(
        cache_hit=False,
        provider_used=provider_name,
        tokens_in=usage.prompt_tokens,
        tokens_out=usage.completion_tokens,
        cost_usd=cost_usd,
    )
    fastapi_response.headers["x-cache-hit"] = "false"
    fastapi_response.headers["x-provider-used"] = provider_name
    fastapi_response.headers["x-routing-tier"] = tier_label
    fastapi_response.headers["x-cost-usd"] = f"{cost_usd:.4f}"
    return completion
