"""Retry with full jitter, inside timeout budgets (DESIGN.md §9).

``max_retries=3`` means 3 TOTAL attempts, matching the design doc. The budget
is one clock per request, created in the route and shared by this layer and
the fallback walker — a slow provider can never eat the whole request because
every consumer asks the same object what's left.

Testability is built in: ``sleep`` and the RNG are injectable, so retry tests
run in milliseconds and jitter bounds are checkable without touching globals.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

from llm_gateway.config import ReliabilitySettings
from llm_gateway.schemas.errors import (
    AttemptTimedOut,
    ProviderFailure,
    ProviderRejected,
    RequestTimedOut,
)


class TimeoutBudget:
    """One clock per request, shared by every layer below the route (§9)."""

    def __init__(self, total_s: float) -> None:
        self._deadline = time.monotonic() + total_s

    @property
    def remaining_s(self) -> float:
        return self._deadline - time.monotonic()

    @property
    def expired(self) -> bool:
        return self.remaining_s <= 0


def full_jitter(attempt: int, settings: ReliabilitySettings, rng: random.Random) -> float:
    """AWS-style full jitter: uniform(0, min(cap, base * 2^(attempt-1))).
    Pure + rng-injected so bounds are testable."""
    ceiling = min(settings.backoff_cap_s, settings.backoff_base_s * (2 ** (attempt - 1)))
    return rng.uniform(0, ceiling)


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    budget: TimeoutBudget,
    settings: ReliabilitySettings,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
) -> T:
    """Run one provider call with up to ``max_retries`` TOTAL attempts.

    ProviderRejected re-raises immediately. Retryable failures back off with
    full jitter — but never sleep past the point where the budget is dead.
    """
    rng = rng or random.Random()
    total_attempts = max(1, settings.max_retries)  # 0 attempts is not a thing
    last_error: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        if budget.expired:
            raise RequestTimedOut(f"request budget exhausted before attempt {attempt}")
        slice_s = min(settings.per_attempt_timeout_s, budget.remaining_s)
        try:
            return await asyncio.wait_for(fn(), timeout=slice_s)
        except ProviderRejected:
            raise  # retrying an identical rejected request is pointless
        except TimeoutError as exc:
            # not retried; yielding to fallback. A per-attempt timeout is not
            # evidence the next slice will land, and re-slicing the same
            # provider is how a slow-but-alive upstream once burned the whole
            # budget while healthy fallbacks never ran (budget starvation,
            # DESIGN.md §9). Raise; the walker spends what's left on the chain.
            raise AttemptTimedOut(
                f"attempt {attempt} exceeded its {slice_s:.2f}s timeout slice"
            ) from exc
        except ProviderFailure as exc:
            last_error = exc
            if attempt == total_attempts:
                break
            pause = full_jitter(attempt, settings, rng)
            if budget.remaining_s - pause <= 0:
                raise RequestTimedOut(
                    f"pausing {pause:.2f}s for retry would exceed the request budget"
                ) from exc
            await sleep(pause)

    assert last_error is not None  # loop ran at least once
    raise ProviderFailure(f"failed after {total_attempts} attempts: {last_error}") from last_error
