"""Anthropic adapter contract — the dialect assertions live here: system
extracted to top level, max_tokens always present, stop_sequences not stop."""

import json
from pathlib import Path

import pytest
import respx

from llm_gateway.config import AnthropicProviderSettings
from llm_gateway.providers.anthropic_adapter import AnthropicAdapter
from llm_gateway.schemas.errors import ProviderFailure, ProviderRejected
from llm_gateway.schemas.openai_api import ChatCompletionRequest, Message

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "anthropic_message.json").read_text())
URL = "https://api.anthropic.com/v1/messages"


def adapter() -> AnthropicAdapter:
    return AnthropicAdapter(AnthropicProviderSettings(api_key="test", _env_file=None))


@respx.mock
async def test_happy_path_dialect_and_parsing():
    route = respx.post(URL).respond(200, json=FIXTURE)
    response = await adapter().complete(
        ChatCompletionRequest(
            model="auto",
            messages=[
                Message(role="system", content="Be terse."),
                Message(role="user", content="hi"),
            ],
            stop=["\n\n"],
        )
    )

    # ── the sent side: the dialect contract ──
    sent = json.loads(route.calls.last.request.content)
    assert sent["system"] == "Be terse."  # extracted, not in messages
    assert all(m["role"] != "system" for m in sent["messages"])
    assert sent["max_tokens"] == 1024  # defaulted — Anthropic requires it
    assert sent["stop_sequences"] == ["\n\n"]  # renamed
    assert "stop" not in sent
    headers = route.calls.last.request.headers
    assert headers["x-api-key"] == "test"
    assert headers["anthropic-version"] == "2023-06-01"

    # ── the returned side: normalized to OpenAI schema ──
    assert response.choices[0].message.content == "Hi there! How can I help?"  # blocks joined
    assert response.choices[0].finish_reason == "stop"  # end_turn → stop
    assert response.usage.prompt_tokens == 8  # input_tokens → prompt_tokens
    assert response.usage.total_tokens == 14


@respx.mock
async def test_real_model_name_passes_through():
    route = respx.post(URL).respond(200, json=FIXTURE)
    await adapter().complete(
        ChatCompletionRequest(
            model="claude-3-5-sonnet-latest", messages=[Message(role="user", content="hi")]
        )
    )
    assert json.loads(route.calls.last.request.content)["model"] == "claude-3-5-sonnet-latest"


@respx.mock
async def test_500_is_a_retryable_failure():
    respx.post(URL).respond(500, json={"type": "error", "error": {"message": "overloaded"}})
    with pytest.raises(ProviderFailure) as exc_info:
        await adapter().complete(
            ChatCompletionRequest(model="auto", messages=[Message(role="user", content="hi")])
        )
    assert exc_info.value.retryable is True


@respx.mock
async def test_400_is_embedded_in_the_rejection():
    respx.post(URL).respond(
        400,
        json={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "max_tokens: field required"},
        },
    )
    with pytest.raises(ProviderRejected) as exc_info:
        await adapter().complete(
            ChatCompletionRequest(model="auto", messages=[Message(role="user", content="hi")])
        )
    assert "field required" in str(exc_info.value)
