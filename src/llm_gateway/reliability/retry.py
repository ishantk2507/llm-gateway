"""Retry with full jitter, inside timeout budgets (DESIGN.md §9).

``max_retries`` means TOTAL attempts. Timeout semantics (the live Day-4
finding, fixed): a per-attempt timeout raises AttemptTimedOut — NOT retried
on the same provider (restarting the same slow work just burns the budget);
the fallback walker catches it and spends the remaining budget on the next
provider. A chain that dies entirely of timeouts is a 504, not a 503.

Budget death BEFORE an attempt, or a backoff sleep that would cross the
deadline, raises plain RequestTimedOut — the walker does NOT catch that one:
a dead budget stops everything.

Testability is built in: ``sleep`` and the RNG are injectable.
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
    rng = rng or random.Random()
    total_attempts = max(1, settings.max_retries)
    last_error: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        if budget.expired:
            raise RequestTimedOut(f"request budget exhausted before attempt {attempt}")
        slice_s = min(settings.per_attempt_timeout_s, budget.remaining_s)
        try:
            return await asyncio.wait_for(fn(), timeout=slice_s)
        except TimeoutError as exc:
            raise AttemptTimedOut(
                f"attempt {attempt} timed out after {slice_s:.2f}s — "
                "yielding to fallback, not retried on this provider"
            ) from exc
        except ProviderRejected:
            raise  # retrying an identical rejected request is pointless
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

    assert last_error is not None
    raise ProviderFailure(f"failed after {total_attempts} attempts: {last_error}") from last_error
