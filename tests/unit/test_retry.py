"""Retry semantics with an injected sleep recorder — millisecond tests,
deterministic assertions."""

import random

import pytest

from llm_gateway.config import ReliabilitySettings
from llm_gateway.reliability.retry import TimeoutBudget, full_jitter, with_retries
from llm_gateway.schemas.errors import ProviderFailure, ProviderRejected, RequestTimedOut


def settings(**overrides) -> ReliabilitySettings:
    base = dict(
        max_retries=3,
        backoff_base_s=0.5,
        backoff_cap_s=5.0,
        per_attempt_timeout_s=10.0,
        request_timeout_budget_s=30.0,
    )
    base.update(overrides)
    return ReliabilitySettings(**base)


class Recorder:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


async def test_fails_twice_then_succeeds_with_jittered_backoff():
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderFailure("boom")
        return "ok"

    recorder = Recorder()
    result = await with_retries(
        flaky,
        budget=TimeoutBudget(30),
        settings=settings(backoff_base_s=0.1, backoff_cap_s=0.2),
        sleep=recorder.sleep,
    )
    assert result == "ok"
    assert calls == 3 and len(recorder.sleeps) == 2
    assert all(0 <= s <= 0.2 for s in recorder.sleeps)  # full jitter bounds hold


async def test_rejected_is_never_retried():
    calls = 0

    async def rejected() -> None:
        nonlocal calls
        calls += 1
        raise ProviderRejected("no")

    recorder = Recorder()
    with pytest.raises(ProviderRejected):
        await with_retries(
            rejected, budget=TimeoutBudget(30), settings=settings(), sleep=recorder.sleep
        )
    assert calls == 1 and recorder.sleeps == []


async def test_exhaustion_raises_provider_failure_with_count():
    async def always_fails() -> None:
        raise ProviderFailure("boom")

    with pytest.raises(ProviderFailure, match="3 attempts"):
        await with_retries(
            always_fails, budget=TimeoutBudget(30), settings=settings(), sleep=Recorder().sleep
        )


async def test_budget_death_between_attempts_is_a_504():
    async def always_fails() -> None:
        raise ProviderFailure("boom")

    with pytest.raises(RequestTimedOut):
        await with_retries(
            always_fails, budget=TimeoutBudget(0.001), settings=settings(), sleep=Recorder().sleep
        )


def test_full_jitter_bounds():
    s = settings(backoff_base_s=0.5, backoff_cap_s=5.0)
    rng = random.Random(42)
    for attempt in range(1, 5):
        ceiling = min(5.0, 0.5 * (2 ** (attempt - 1)))
        assert all(0 <= full_jitter(attempt, s, rng) <= ceiling for _ in range(200))
