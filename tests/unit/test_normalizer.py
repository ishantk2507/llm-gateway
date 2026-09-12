"""Normalizer — dicts in, models/exceptions out. Zero mocking required;
that's the payoff of concentrating all translation in one module."""

import pytest

from llm_gateway.providers.normalizer import (
    build_anthropic_body,
    build_openai_body,
    classify_provider_error,
    parse_anthropic_response,
)
from llm_gateway.schemas.errors import ProviderFailure, ProviderRejected
from llm_gateway.schemas.openai_api import ChatCompletionRequest, Message


def req(**overrides) -> ChatCompletionRequest:
    base = {"model": "auto", "messages": [Message(role="user", content="hi")]}
    base.update(overrides)
    return ChatCompletionRequest(**base)


def test_openai_body_is_exactly_the_whitelist():
    body = build_openai_body(req(), "gpt-4o-mini")
    assert set(body) == {"model", "messages"}
    assert body["model"] == "gpt-4o-mini"  # virtual 'auto' → configured default


def test_openai_body_includes_only_set_params():
    body = build_openai_body(req(temperature=0.2, max_tokens=100), "m")
    assert body["temperature"] == 0.2 and body["max_tokens"] == 100
    assert "top_p" not in body and "stop" not in body


def test_anthropic_body_extracts_system_and_defaults_max_tokens():
    body = build_anthropic_body(
        req(
            messages=[
                Message(role="system", content="Be terse."),
                Message(role="user", content="hi"),
            ],
            stop=["END"],
        ),
        "claude-3-5-haiku-latest",
    )
    assert body["system"] == "Be terse."
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["max_tokens"] == 1024  # Anthropic requires it; OpenAI doesn't send it
    assert body["stop_sequences"] == ["END"]  # renamed, never 'stop'
    assert "stop" not in body


def test_anthropic_body_keeps_explicit_max_tokens():
    assert build_anthropic_body(req(max_tokens=77), "m")["max_tokens"] == 77


def test_system_only_conversation_is_rejected_not_mangled():
    with pytest.raises(ProviderRejected):
        build_anthropic_body(req(messages=[Message(role="system", content="s")]), "m")


def test_anthropic_response_parsing():
    payload = {
        "id": "msg_1",
        "model": "claude-3-5-haiku",
        "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }
    r = parse_anthropic_response(payload, "fallback")
    assert r.choices[0].message.content == "ab"  # blocks joined
    assert r.choices[0].finish_reason == "length"  # max_tokens → length
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens) == (3, 4, 7)


def test_error_classification():
    with pytest.raises(ProviderRejected) as rejected:
        classify_provider_error(status_code=400, provider="openai", body="context too long")
    assert rejected.value.retryable is False

    with pytest.raises(ProviderFailure) as limited:  # 429 is transient, not a rejection
        classify_provider_error(status_code=429, provider="openai", body="slow down")
    assert limited.value.retryable is True

    with pytest.raises(ProviderFailure):
        classify_provider_error(status_code=503, provider="openai", body="down")
