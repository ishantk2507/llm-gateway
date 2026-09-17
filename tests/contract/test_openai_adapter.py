"""OpenAI adapter contract — respx intercepts httpx and replays a committed
fixture. Zero network, zero quota, forever (ADR-0006)."""

import json
from pathlib import Path

import pytest
import respx

from llm_gateway.config import OpenAIProviderSettings
from llm_gateway.providers.openai_adapter import OpenAIAdapter
from llm_gateway.schemas.errors import ProviderFailure, ProviderRejected
from llm_gateway.schemas.openai_api import ChatCompletionRequest, Message

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "openai_chat_completion.json").read_text()
)
URL = "https://api.openai.com/v1/chat/completions"


def adapter() -> OpenAIAdapter:
    return OpenAIAdapter(OpenAIProviderSettings(api_key="sk-test", _env_file=None))


def req(**overrides) -> ChatCompletionRequest:
    base = {"model": "auto", "messages": [Message(role="user", content="hi")]}
    base.update(overrides)
    return ChatCompletionRequest(**base)


@respx.mock
async def test_happy_path_shape_and_virtual_model_substitution():
    route = respx.post(URL).respond(200, json=FIXTURE)
    response = await adapter().complete(req())

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 16

    sent = json.loads(route.calls.last.request.content)  # the whitelist, tested from outside
    assert sent["model"] == "gpt-4o-mini"  # virtual 'auto' → configured default
    assert set(sent) == {"model", "messages"}  # no null params ever leave
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-test"


@respx.mock
async def test_real_model_name_passes_through():
    route = respx.post(URL).respond(200, json=FIXTURE)
    await adapter().complete(req(model="gpt-4o"))
    assert json.loads(route.calls.last.request.content)["model"] == "gpt-4o"


@respx.mock
async def test_500_is_a_retryable_failure():
    respx.post(URL).respond(500, json={"error": {"message": "upstream melted"}})
    with pytest.raises(ProviderFailure) as exc_info:
        await adapter().complete(req())
    assert exc_info.value.retryable is True


@respx.mock
async def test_400_is_embedded_in_the_rejection():
    respx.post(URL).respond(
        400, json={"error": {"message": "context length exceeded", "type": "invalid_request_error"}}
    )
    with pytest.raises(ProviderRejected) as exc_info:
        await adapter().complete(req())
    assert "context length exceeded" in str(exc_info.value)  # provider's own words preserved
    assert exc_info.value.retryable is False


@respx.mock
async def test_failure_messages_name_the_provider():
    respx.post(URL).respond(500, json={"error": {"message": "melted"}})
    with pytest.raises(ProviderFailure, match="^openai"):
        await adapter().complete(req())
