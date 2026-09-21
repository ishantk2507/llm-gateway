"""Circuit breakers (DESIGN.md §9) — per provider, request-outcome derived.

This is the fix for the live Day-4/5 finding: a persistently failing provider
(the Gemini 503 session) was retried on EVERY request — the "gemini tax."
Once tripped, the breaker skips the provider for the cooldown instead of
re-failing per request.

States: CLOSED → OPEN → HALF_OPEN.
- CLOSED: tracks the last ``breaker_window`` TERMINAL outcomes (one sample
  per with_retries conclusion, not per attempt). Window FULL and failure
  share >= ``breaker_failure_rate`` → OPEN.
- OPEN: ``allow()`` refuses for ``breaker_cooldown_s``, then lazily → HALF_OPEN.
- HALF_OPEN: admits up to ``breaker_half_open_probes`` probes. First probe
  success → CLOSED (window cleared). Probe failure → OPEN with fresh cooldown.

Stale outcomes are dropped BY CONSTRUCTION via a generation (epoch) counter:
``allow()`` returns the admission epoch; ``record_success``/``record_failure``
no-op on epoch mismatch. An outcome from before a transition can never
contaminate the new state's accounting — including the two-probes-racing edge
(probe 2's outcome arriving after probe 1 already closed the breaker).

Outcome mapping is decided by the FALLBACK WALKER, not the breaker:
ProviderFailure / AttemptTimedOut → failure; completion AND ProviderRejected
→ success. A coherent rejection (4xx) means the provider is alive and
processing — breakers measure HEALTH, not request quality.

Accepted consequence, stated verbatim: a provider with an invalid API key is
rejected coherently in ~100ms per request and NEVER trips its breaker.
Correct: fast coherent failures don't need breaker protection — the chain
moves on immediately (RUNBOOK row exists for this exact symptom).

Pinned-provider semantics: a request pinned to a provider whose breaker is
OPEN gets an explicit 503 naming the circuit. Deliberate — the client chose
that provider; hammering it request-after-request is worse than a visible
error.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from enum import IntEnum

from llm_gateway.config import ReliabilitySettings


class BreakerState(IntEnum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


class CircuitBreaker:
    def __init__(
        self,
        *,
        window: int,
        failure_rate: float,
        cooldown_s: float,
        half_open_probes: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window: deque[bool] = deque(maxlen=window)  # True marks a FAILURE sample
        self._failure_rate = failure_rate
        self._cooldown_s = cooldown_s
        self._half_open_probes = half_open_probes
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._epoch = 0
        self._opened_at = 0.0
        self._probes_admitted = 0

    @property
    def state(self) -> BreakerState:
        # Lazy OPEN → HALF_OPEN: the transition happens on read, once the
        # cooldown has fully elapsed. (Mutating property — documented.)
        if (
            self._state is BreakerState.OPEN
            and (self._clock() - self._opened_at) >= self._cooldown_s
        ):
            self._state = BreakerState.HALF_OPEN
            self._epoch += 1
            self._probes_admitted = 0
        return self._state

    def allow(self) -> int | None:
        """Admission decision → the admission epoch, or None if denied."""
        state = self.state
        if state is BreakerState.CLOSED:
            return self._epoch
        if state is BreakerState.HALF_OPEN:
            if self._probes_admitted >= self._half_open_probes:
                return None
            self._probes_admitted += 1
            return self._epoch
        return None  # OPEN

    def record_success(self, epoch: int) -> None:
        if epoch != self._epoch:
            return  # stale outcome — dropped by construction
        if self._state is BreakerState.HALF_OPEN:
            self._close()
        elif self._state is BreakerState.CLOSED:
            self._window.append(False)

    def record_failure(self, epoch: int) -> None:
        if epoch != self._epoch:
            return
        if self._state is BreakerState.HALF_OPEN:
            self._trip()
        elif self._state is BreakerState.CLOSED:
            self._window.append(True)
            if (
                len(self._window) == self._window.maxlen
                and sum(self._window) / len(self._window) >= self._failure_rate
            ):
                self._trip()

    def snapshot(self) -> str:
        return self.state.name.lower()

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        self._epoch += 1
        self._window.clear()
        self._probes_admitted = 0

    def _close(self) -> None:
        self._state = BreakerState.CLOSED
        self._epoch += 1
        self._window.clear()
        self._probes_admitted = 0


class BreakerBoard:
    """One breaker per provider name, created on first use. Injectable clock
    for deterministic tests; optional on_state callback (main wires it to the
    Prometheus gauge)."""

    def __init__(
        self,
        settings: ReliabilitySettings,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_state: Callable[[str, int], None] | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._on_state = on_state or (lambda name, state: None)
        self._breakers: dict[str, CircuitBreaker] = {}

    def breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                window=self._settings.breaker_window,
                failure_rate=self._settings.breaker_failure_rate,
                cooldown_s=self._settings.breaker_cooldown_s,
                half_open_probes=self._settings.breaker_half_open_probes,
                clock=self._clock,
            )
        return self._breakers[name]

    def allow(self, name: str) -> int | None:
        breaker = self.breaker(name)
        epoch = breaker.allow()
        self._on_state(name, int(breaker.state))
        return epoch

    def record_success(self, name: str, epoch: int) -> None:
        breaker = self.breaker(name)
        breaker.record_success(epoch)
        self._on_state(name, int(breaker.state))

    def record_failure(self, name: str, epoch: int) -> None:
        breaker = self.breaker(name)
        breaker.record_failure(epoch)
        self._on_state(name, int(breaker.state))

    def snapshot(self) -> dict[str, str]:
        return {name: breaker.snapshot() for name, breaker in self._breakers.items()}
