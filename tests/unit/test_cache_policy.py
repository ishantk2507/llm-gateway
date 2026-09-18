"""Policy — pure functions, instant tests, no I/O."""

import pytest

from llm_gateway.cache import policy
from llm_gateway.schemas.openai_api import ChatCompletionRequest, Message


def r(**overrides) -> ChatCompletionRequest:
    base = {"model": "auto", "messages": [Message(role="user", content="hi")]}
    base.update(overrides)
    return ChatCompletionRequest(**base)


def test_single_turn_is_cacheable():
    assert policy.is_cacheable(r())
    assert policy.is_cacheable(
        r(messages=[Message(role="system", content="s"), Message(role="user", content="hi")])
    )


def test_multi_turn_is_never_cacheable():
    assert not policy.is_cacheable(
        r(
            messages=[
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
                Message(role="user", content="again"),
            ]
        )
    )


@pytest.mark.parametrize(
    "temperature,expected", [(None, True), (0.0, True), (0.3, True), (0.4, False), (1.7, False)]
)
def test_temperature_gate(temperature, expected):
    assert policy.is_cacheable(r(temperature=temperature)) is expected


def test_tool_requests_are_never_cacheable():
    assert not policy.is_cacheable(r(tools=[{"type": "function", "function": {"name": "x"}}]))
    assert not policy.is_cacheable(r(tool_choice="auto"))


def test_streaming_is_never_cacheable():
    assert not policy.is_cacheable(r(stream=True))


def test_normalized_prompt_collapses_whitespace():
    request = r(messages=[Message(role="user", content="  what   is\n\n  it?  ")])
    assert policy.normalized_prompt(request) == "what is it?"


@pytest.mark.parametrize(
    "model,family",
    [
        ("gpt-4o", "openai"),
        ("o3-mini", "openai"),
        ("claude-3-5-haiku-latest", "anthropic"),
        ("auto", "routed"),
        ("premium", "routed"),
        ("mock", "mock"),
        ("llama-3", "other"),
        ("gemini-2.0-flash", "gemini"),
    ],
)
def test_model_family_mapping(model, family):
    assert policy.model_family(model) == family


def test_prompt_hash_is_deterministic_and_family_scoped():
    h1 = policy.prompt_hash("hi", "openai", "low")
    assert h1 == policy.prompt_hash("hi", "openai", "low")
    assert h1 != policy.prompt_hash("hi", "anthropic", "low")
