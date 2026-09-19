"""Phase 3 acceptance — routing decisions visible in headers and logs, through
the real app with the committed rule/mapping data (deterministic, no network)."""

from fastapi.testclient import TestClient
from tests.helpers import hermetic_settings

from llm_gateway.config import (
    AppSettings,
    Environment,
    MockSettings,
    ReliabilitySettings,
    Settings,
)
from llm_gateway.main import create_app
from llm_gateway.providers.base import ProviderRegistry
from llm_gateway.providers.mock_adapter import MockProvider

HELLO = {"model": "auto", "messages": [{"role": "user", "content": "Hello, gateway."}]}
EXPLAIN = {
    "model": "auto",
    "messages": [
        {"role": "user", "content": "Explain the circuit breaker pattern in distributed systems."}
    ],
}
CODE = {
    "model": "auto",
    "messages": [
        {"role": "user", "content": "Write a Python function that reverses a linked list."}
    ],
}


def make_settings(**mock_overrides) -> Settings:
    return hermetic_settings(
        app=AppSettings(environment=Environment.TEST),
        mock=MockSettings(latency_ms=1, **mock_overrides),
        reliability=ReliabilitySettings(backoff_base_s=0.001, backoff_cap_s=0.002),
    )


def client_for(settings: Settings, providers: ProviderRegistry | None = None) -> TestClient:
    return TestClient(create_app(settings, providers=providers))


def test_greeting_routes_cheap_with_rule_in_headers():
    with client_for(make_settings()) as client:
        r = client.post("/v1/chat/completions", json=HELLO)
    assert r.status_code == 200
    assert r.headers["x-routing-tier"] == "cheap"


def test_explain_routes_standard():
    with client_for(make_settings()) as client:
        r = client.post("/v1/chat/completions", json=EXPLAIN)
    assert r.headers["x-routing-tier"] == "standard"


def test_code_routes_premium():
    with client_for(make_settings()) as client:
        r = client.post("/v1/chat/completions", json=CODE)
    assert r.headers["x-routing-tier"] == "premium"


def test_header_override_pins_the_tier():
    with client_for(make_settings()) as client:
        r = client.post(
            "/v1/chat/completions", json=HELLO, headers={"x-routing-strategy": "premium"}
        )
    assert r.headers["x-routing-tier"] == "premium"


def test_model_pin_goes_to_that_provider():
    registry = ProviderRegistry()
    registry.register(MockProvider(MockSettings(latency_ms=1), name="openai"))
    registry.register(MockProvider(MockSettings(latency_ms=1)))
    with client_for(make_settings(), providers=registry) as client:
        r = client.post("/v1/chat/completions", json={**HELLO, "model": "gpt-4o"})
    assert r.status_code == 200
    assert r.headers["x-provider-used"] == "openai"
    assert r.headers["x-routing-tier"] == "pinned"


def test_pinned_provider_failure_is_still_a_502():
    registry = ProviderRegistry()
    registry.register(MockProvider(MockSettings(latency_ms=0, error_rate=1.0), name="openai"))
    with client_for(make_settings(), providers=registry) as client:
        r = client.post("/v1/chat/completions", json={**HELLO, "model": "gpt-4o"})
    assert r.status_code == 502  # pin semantics preserved from Day 2


def test_unconfigured_pin_routes_instead_of_failing():
    with client_for(make_settings()) as client:  # mock-only registry
        r = client.post("/v1/chat/completions", json={**HELLO, "model": "gpt-4o"})
    assert r.status_code == 200  # warned, then routed
    assert r.headers["x-provider-used"] == "mock"


def test_gemini_model_pin_routes_to_gemini_provider():
    registry = ProviderRegistry()
    registry.register(MockProvider(MockSettings(latency_ms=1), name="gemini"))
    registry.register(MockProvider(MockSettings(latency_ms=1)))
    with client_for(make_settings(), providers=registry) as client:
        r = client.post("/v1/chat/completions", json={**HELLO, "model": "gemini-2.0-flash"})
    assert r.status_code == 200
    assert r.headers["x-provider-used"] == "gemini"
    assert r.headers["x-routing-tier"] == "pinned"
