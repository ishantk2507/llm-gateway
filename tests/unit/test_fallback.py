"""Chain-walker semantics with fakes — above the HTTP layer, so no respx."""

import asyncio

import pytest

from llm_gateway.config import ReliabilitySettings
from llm_gateway.providers.base import ProviderAdapter
from llm_gateway.reliability.fallback import execute_with_fallback
from llm_gateway.reliability.retry import TimeoutBudget
from llm_gateway.schemas.errors import (
    AllProvidersDown,
    ProviderFailure,
    ProviderRejected,
    RequestTimedOut,
)
from llm_gateway.schemas.openai_api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)


def one_shot() -> ReliabilitySettings:
    """One attempt per provider — tests the WALKER, not the retry layer."""
    return ReliabilitySettings(
        max_retries=1,
        backoff_base_s=0.001,
        backoff_cap_s=0.002,
        per_attempt_timeout_s=5.0,
        request_timeout_budget_s=10.0,
    )


def req() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="auto", messages=[Message(role="user", content="q")])


def resp() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="x",
        created=1,
        model="m",
        choices=[Choice(message=Message(role="assistant", content="r"))],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class FakeAdapter(ProviderAdapter):
    def __init__(self, name: str, script: list) -> None:
        self.name = name
        self.script = list(script)
        self.calls = 0

    async def complete(self, request) -> ChatCompletionResponse:
        self.calls += 1
        item = self.script.pop(0) if self.script else ProviderFailure("script exhausted")
        if isinstance(item, Exception):
            raise item
        return item

    async def health_check(self) -> bool:
        return True


class SlowBrokenAdapter(ProviderAdapter):
    name = "slow"

    async def complete(self, request) -> ChatCompletionResponse:
        await asyncio.sleep(0.05)
        raise ProviderFailure("slow death")

    async def health_check(self) -> bool:
        return True


async def test_first_healthy_provider_serves():
    a, b = FakeAdapter("a", [resp()]), FakeAdapter("b", [resp()])
    _, name = await execute_with_fallback(
        [a, b], req(), budget=TimeoutBudget(10), settings=one_shot()
    )
    assert name == "a" and b.calls == 0


async def test_failure_fails_over_in_order():
    a, b = FakeAdapter("a", [ProviderFailure("down")]), FakeAdapter("b", [resp()])
    _, name = await execute_with_fallback(
        [a, b], req(), budget=TimeoutBudget(10), settings=one_shot()
    )
    assert name == "b" and a.calls == 1


async def test_rejection_also_moves_on():
    a, b = FakeAdapter("a", [ProviderRejected("too long")]), FakeAdapter("b", [resp()])
    _, name = await execute_with_fallback(
        [a, b], req(), budget=TimeoutBudget(10), settings=one_shot()
    )
    assert name == "b"


async def test_chain_exhaustion_raises_all_providers_down():
    a, b = FakeAdapter("a", [ProviderFailure("x")]), FakeAdapter("b", [ProviderFailure("y")])
    with pytest.raises(AllProvidersDown, match="a, b"):
        await execute_with_fallback([a, b], req(), budget=TimeoutBudget(10), settings=one_shot())


async def test_pinned_single_failure_escalates_as_upstream_error():
    a = FakeAdapter("a", [ProviderFailure("down")])
    with pytest.raises(ProviderFailure):  # escapes as 502, not 503 (DESIGN.md §7.3)
        await execute_with_fallback([a], req(), budget=TimeoutBudget(10), settings=one_shot())


async def test_budget_death_between_providers_is_a_504():
    healthy = FakeAdapter("b", [resp()])
    with pytest.raises(RequestTimedOut):
        await execute_with_fallback(
            [SlowBrokenAdapter(), healthy],
            req(),
            budget=TimeoutBudget(0.02),
            settings=one_shot(),
        )
    assert healthy.calls == 0  # never started — the budget said no
