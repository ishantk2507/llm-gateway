"""Ordered chain walker (DESIGN.md §9) with circuit-breaker integration.

Failure semantics (unchanged): pinned-single final failure → 502; routed
chain exhausted → 503; budget death → 504.

Breaker integration (Day 6):
- A provider whose breaker refuses admission is SKIPPED — logged, no calls,
  no window samples (skipping is not an outcome).
- Every with_retries CONCLUSION maps to exactly ONE breaker outcome:
  ProviderFailure / AttemptTimedOut → failure; completion and
  ProviderRejected → success (coherent rejection = alive provider).
- All providers skipped (every circuit open) → AllProvidersDown 503 naming
  the circuits — better a visible 503 than hammering dead upstreams.
  Pinned + open circuit → the same 503, deliberately (RUNBOOK row).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from llm_gateway.config import ReliabilitySettings
from llm_gateway.observability.logging import get_logger
from llm_gateway.providers.base import ProviderAdapter
from llm_gateway.reliability.circuit_breaker import BreakerBoard
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
    breakers: BreakerBoard | None = None,
) -> tuple[ChatCompletionResponse, str]:
    if not chain:
        raise AllProvidersDown("empty provider chain")  # defensive; config forbids this

    last_error: ProviderFailure | ProviderRejected | AttemptTimedOut | None = None
    tried_any = False

    for index, provider in enumerate(chain):
        if budget.expired:
            raise RequestTimedOut("request budget exhausted before trying the next provider")

        epoch = breakers.allow(provider.name) if breakers is not None else 0
        if breakers is not None and epoch is None:
            logger.warning(
                "breaker_open_skipping",
                provider=provider.name,
                providers_remaining=len(chain) - index - 1,
            )
            continue

        tried_any = True
        try:
            response = await with_retries(
                partial(provider.complete, request), budget=budget, settings=settings
            )
        except (ProviderFailure, ProviderRejected, AttemptTimedOut) as exc:
            # plain RequestTimedOut deliberately NOT caught: dead budget stops everything
            last_error = exc
            if breakers is not None:
                if isinstance(exc, ProviderRejected):
                    breakers.record_success(provider.name, epoch)  # coherent — provider alive
                else:
                    breakers.record_failure(provider.name, epoch)
            logger.warning(
                "provider_failed_moving_on",
                provider=provider.name,
                error=str(exc),
                providers_remaining=len(chain) - index - 1,
            )
            continue

        if breakers is not None:
            breakers.record_success(provider.name, epoch)
        return response, provider.name

    if not tried_any:
        raise AllProvidersDown(
            f"all {len(chain)} provider circuits open: {', '.join(p.name for p in chain)}"
        )
    if budget.remaining_s < settings.per_attempt_timeout_s:
        # Budget can't fund another attempt anywhere — timeout, not availability
        raise RequestTimedOut(f"budget exhausted after {len(chain)} provider(s)")
    if len(chain) == 1 and last_error is not None:
        raise last_error  # pinned: upstream error (502), not an availability claim
    raise AllProvidersDown(f"all {len(chain)} providers failed: {', '.join(p.name for p in chain)}")
