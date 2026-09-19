"""Phase 1 acceptance: failover, exhaustion, pinned failure, budgets, retries —
through the real app, all offline (ADR-0006)."""

import asyncio

from fastapi.testclient import TestClient

from llm_gateway.config import (
    AppSettings,
    Environment,
    MockSettings,
    ReliabilitySettings,
    Settings,
)
from llm_gateway.main import create_app
from llm_gateway.providers.base import ProviderAdapter, ProviderRegistry
from llm_gateway.providers.mock_adapter import MockProvider
from llm_gateway.schemas.openai_api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

HELLO = {"model": "auto", "messages": [{"role": "user", "content": "Hello, gateway."}]}


def make_settings(**reliability_overrides) -> Settings:
    base = dict(
        max_retries=3,
        backoff_base_s=0.001,
        backoff_cap_s=0.002,
        per_attempt_timeout_s=5.0,
        request_timeout_budget_s=10.0,
    )
    base.update(reliability_overrides)
    return Settings(
        _env_file=None,
        app=AppSettings(environment=Environment.TEST),
        reliability=ReliabilitySettings(**base),
    )


def registry_with(*adapters: ProviderAdapter) -> ProviderRegistry:
    registry = ProviderRegistry()
    for a in adapters:
        registry.register(a)
    return registry


def client_for(settings: Settings, registry: ProviderRegistry) -> TestClient:
    return TestClient(create_app(settings, providers=registry))


class ScriptedAdapter(ProviderAdapter):
    """Fails N times, then succeeds — proves retries happen through the app."""

    def __init__(self, name: str, failures: int) -> None:
        self.name = name
        self.failures = failures
        self.calls = 0

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise _failure(f"scripted failure {self.calls}")
        return ChatCompletionResponse(
            id="chatcmpl-scripted",
            created=1,
            model="scripted",
            choices=[Choice(message=Message(role="assistant", content="made it"))],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def health_check(self) -> bool:
        return True


def _failure(msg: str):  # local import dodge at module top
    from llm_gateway.schemas.errors import ProviderFailure

    return ProviderFailure(msg)


def test_primary_failure_fails_over_transparently():
    settings = make_settings()
    primary = MockProvider(MockSettings(latency_ms=0, error_rate=1.0), name="primary")
    backup = MockProvider(MockSettings(latency_ms=1), name="backup")
    with client_for(settings, registry_with(primary, backup)) as client:
        response = client.post("/v1/chat/completions", json=HELLO)

    assert response.status_code == 200
    assert response.headers["x-provider-used"] == "backup"  # who actually served


def test_every_provider_down_fails_loud_with_503():
    settings = make_settings()
    dead_a = MockProvider(MockSettings(latency_ms=0, error_rate=1.0), name="a")
    dead_b = MockProvider(MockSettings(latency_ms=0, error_rate=1.0), name="b")
    with client_for(settings, registry_with(dead_a, dead_b)) as client:
        response = client.post("/v1/chat/completions", json=HELLO)

    assert response.status_code == 503  # ADR-0004: explicit, never a stale guess
    error = response.json()["error"]
    assert error["type"] == "service_unavailable"
    assert "a, b" in error["message"]


def test_pinned_provider_failure_is_a_502():
    settings = make_settings()
    dead = MockProvider(MockSettings(latency_ms=0, error_rate=1.0))  # name "mock"
    with client_for(settings, registry_with(dead)) as client:
        response = client.post("/v1/chat/completions", json={**HELLO, "model": "mock"})

    assert response.status_code == 502  # pinned: upstream error, not availability
    assert response.json()["error"]["type"] == "api_error"


def test_dead_budget_is_a_504():
    # max_retries >= 2 is a correctness precondition: with one attempt, no
    # single attempt can consume the whole budget (slice = min(per_attempt,
    # remaining) <= per_attempt < budget), so the walker would reach provider
    # b and the outcome would be a 503. The 504 must fire in the between-
    # attempt budget checks. Mocks at 1000ms >> every possible slice.
    settings = make_settings(
        max_retries=3,
        per_attempt_timeout_s=0.1,
        request_timeout_budget_s=0.15,
    )
    slow_a = MockProvider(MockSettings(latency_ms=1000), name="a")
    slow_b = MockProvider(MockSettings(latency_ms=1000), name="b")
    with client_for(settings, registry_with(slow_a, slow_b)) as client:
        response = client.post("/v1/chat/completions", json=HELLO)

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "timeout"


def test_retries_actually_happen_through_the_app():
    settings = make_settings()
    flaky = ScriptedAdapter("flaky", failures=2)
    with client_for(settings, registry_with(flaky)) as client:
        response = client.post("/v1/chat/completions", json=HELLO)

    assert response.status_code == 200
    assert flaky.calls == 3  # failed twice, third attempt served it


class CountingMock(ProviderAdapter):
    """Slow-but-alive upstream: counts calls, and every call takes longer
    than the attempt slice. The stand-in for the budget-starvation incident's
    Ollama: healthy enough to accept, too slow to answer in time."""

    def __init__(self, name: str, latency_s: float) -> None:
        self.name = name
        self.latency_s = latency_s
        self.calls = 0

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.calls += 1
        await asyncio.sleep(self.latency_s)
        return ChatCompletionResponse(
            id="chatcmpl-slow",
            created=1,
            model="slow",
            choices=[Choice(message=Message(role="assistant", content="finally"))],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def health_check(self) -> bool:
        return True


def test_attempt_timeout_yields_budget_to_fallback_instead_of_retrying():
    """Budget-starvation regression (DESIGN.md §9): a slow primary used to
    burn every attempt slice itself (3 × per_attempt) so the healthy fallback
    never ran and the client got a 504. A per-attempt timeout must NOT be
    retried on the same provider — the remaining budget belongs to the chain."""
    settings = make_settings(
        max_retries=3,  # the old behavior would have used all three on `slow`
        per_attempt_timeout_s=0.05,
        request_timeout_budget_s=1.0,
    )
    slow = CountingMock("slow", latency_s=0.5)  # 10× the attempt slice
    fast = MockProvider(MockSettings(latency_ms=1), name="fast")
    with client_for(settings, registry_with(slow, fast)) as client:
        response = client.post("/v1/chat/completions", json=HELLO)

    assert response.status_code == 200
    assert response.headers["x-provider-used"] == "fast"
    assert slow.calls == 1  # one slice, then the chain moves on — no retry
