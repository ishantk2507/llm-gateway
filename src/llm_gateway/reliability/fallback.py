"""Ordered chain walker (DESIGN.md §9) — the layer that makes provider death
invisible to clients, and the first raiser of AllProvidersDown (ADR-0004).

Failure semantics fall out of chain length:
- pinned (one provider): its final ProviderFailure escapes → 502
  "this upstream errored after its retries"
- routed (whole chain burned): AllProvidersDown → 503, fail loud
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from llm_gateway.config import ReliabilitySettings
from llm_gateway.observability.logging import get_logger
from llm_gateway.providers.base import ProviderAdapter
from llm_gateway.reliability.retry import TimeoutBudget, with_retries
from llm_gateway.schemas.errors import (
    AllProvidersDown,
    AttemptTimedOut,
    ProviderFailure,
    ProviderRejected,
    RequestTimedOut,
)
from llm_gateway.schemas.openai_api import ChatCompletionRequest, ChatCompletionResponse

logger = get_logger(__name__)


async def execute_with_fallback(
    chain: Sequence[ProviderAdapter],
    request: ChatCompletionRequest,
    *,
    budget: TimeoutBudget,
    settings: ReliabilitySettings,
) -> tuple[ChatCompletionResponse, str]:
    """Try each provider in order; return (response, provider_that_served)."""
    if not chain:
        raise AllProvidersDown("empty provider chain")  # defensive; config forbids this

    last_error: ProviderFailure | ProviderRejected | AttemptTimedOut | None = None
    for index, provider in enumerate(chain):
        if budget.expired:
            # No point starting provider #3 with negative time left (§9).
            raise RequestTimedOut("request budget exhausted before trying the next provider")
        try:
            response = await with_retries(
                partial(provider.complete, request),
                budget=budget,
                settings=settings,
            )
        except AttemptTimedOut as exc:
            # One attempt slice expired but the budget may still be alive —
            # that IS a "try the next provider" signal (the budget-starvation
            # fix). A dead budget re-raises: RequestTimedOut is never a
            # move-on signal. Plain RequestTimedOut is deliberately not
            # caught here at all.
            if budget.expired:
                raise
            last_error = exc
            logger.warning(
                "provider_timed_out_moving_on",
                provider=provider.name,
                error=str(exc),
                providers_remaining=len(chain) - index - 1,
                budget_remaining_s=round(budget.remaining_s, 3),
            )
            continue
        except (ProviderFailure, ProviderRejected) as exc:
            # RequestTimedOut deliberately NOT caught — a dead budget is a 504,
            # not a "try the next provider" signal.
            last_error = exc
            logger.warning(
                "provider_failed_moving_on",
                provider=provider.name,
                error=str(exc),
                providers_remaining=len(chain) - index - 1,
            )
            continue
        return response, provider.name

    if len(chain) == 1 and last_error is not None:
        # Pinned provider: the client chose this one; its failure is a plain
        # upstream error (502), its slowness a plain timeout (504).
        raise last_error
    if isinstance(last_error, AttemptTimedOut):
        # The clock killed the chain, not the providers — the client waited,
        # so 504, never a 503 availability claim. Checked by type, not by
        # budget.expired: the last slice can expire with timer epsilon left.
        raise RequestTimedOut(
            f"all {len(chain)} providers exceeded their per-attempt timeout slices"
        ) from last_error
    raise AllProvidersDown(f"all {len(chain)} providers failed: {', '.join(p.name for p in chain)}")
