"""Phase 0 acceptance: the vertical slice through the real app + MockProvider.

No internal mocking — the app is built with the same factory ``make run``
uses, only with test-shaped settings. These tests are the day's "done when":
OpenAI-schema compatibility, determinism, error-envelope shape.
"""

from fastapi.testclient import TestClient

from llm_gateway.config import (
    AppSettings,
    Environment,
    MockSettings,
    ReliabilitySettings,
    Settings,
)
from llm_gateway.main import create_app
from llm_gateway.schemas.openai_api import ChatCompletionResponse

HELLO = {"model": "auto", "messages": [{"role": "user", "content": "Hello, gateway."}]}


def make_settings(**mock_overrides) -> Settings:
    return Settings(
        _env_file=None,
        app=AppSettings(environment=Environment.TEST),
        mock=MockSettings(latency_ms=1, **mock_overrides),
        reliability=ReliabilitySettings(backoff_base_s=0.001, backoff_cap_s=0.002),  # NEW
    )


def client_for(settings: Settings) -> TestClient:
    # Context manager = lifespan runs = registry is built (as in production).
    return TestClient(create_app(settings))


def test_completion_matches_openai_schema_with_contract_headers():
    with client_for(make_settings()) as client:
        response = client.post("/v1/chat/completions", json=HELLO)

    assert response.status_code == 200
    body = response.json()
    ChatCompletionResponse.model_validate(body)  # the compatibility claim, tested
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] >= 1

    assert response.headers["x-provider-used"] == "mock"
    assert response.headers["x-cache-hit"] == "false"
    assert response.headers["x-routing-tier"] == "auto"
    assert response.headers["x-cost-usd"] == "0.0000"


def test_identical_requests_produce_identical_responses():
    with client_for(make_settings()) as client:
        first = client.post("/v1/chat/completions", json=HELLO)
        second = client.post("/v1/chat/completions", json=HELLO)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()  # Day 3's cache tests build on this


def test_stream_true_is_a_400_envelope_not_a_crash():
    with client_for(make_settings()) as client:
        response = client.post("/v1/chat/completions", json={**HELLO, "stream": True})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "stream" in error["message"]


def test_validation_errors_are_openai_400s_not_fastapi_422s():
    with client_for(make_settings()) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto"})

    assert response.status_code == 400  # not 422 — the envelope compatibility fix
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "messages" in error["message"]


def test_multimodal_content_is_rejected_clearly():
    with client_for(make_settings()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            },
        )

    assert response.status_code == 400
    assert "content" in response.json()["error"]["message"]


def test_health_reports_the_mock():
    with client_for(make_settings()) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert {"name": "mock", "healthy": True} in body["providers"]


def test_provider_failure_is_a_502_envelope():
    with client_for(make_settings(error_rate=1.0)) as client:
        response = client.post("/v1/chat/completions", json=HELLO)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"


def test_timeout_budget_exceeded_is_a_504():
    settings = Settings(
        _env_file=None,
        app=AppSettings(environment=Environment.TEST),
        mock=MockSettings(latency_ms=200),
        reliability=ReliabilitySettings(
            backoff_base_s=0.001,
            backoff_cap_s=0.002,
            per_attempt_timeout_s=0.05,
            request_timeout_budget_s=0.06,  # CHANGED (was 1.0): barely over one attempt slice,
            # so the between-attempt budget check fires the 504
        ),
    )
    with client_for(settings) as client:
        response = client.post("/v1/chat/completions", json=HELLO)

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "timeout"
