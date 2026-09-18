"""Gemini adapter contract — the dialect assertions live here: system →
systemInstruction, assistant → model, parts structure, generationConfig
mapping, model in the URL not the body. respx, offline, forever."""

import json
from pathlib import Path

import pytest
import respx

from llm_gateway.config import GeminiProviderSettings
from llm_gateway.providers.gemini_adapter import GeminiAdapter
from llm_gateway.schemas.errors import ProviderFailure, ProviderRejected
from llm_gateway.schemas.openai_api import ChatCompletionRequest, Message

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "gemini_response.json").read_text()
)
BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def adapter() -> GeminiAdapter:
    return GeminiAdapter(GeminiProviderSettings(api_key="test-key", _env_file=None))


def req(**overrides) -> ChatCompletionRequest:
    base = {
        "model": "auto",
        "messages": [
            Message(role="system", content="Be terse."),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="user", content="continue"),
        ],
    }
    base.update(overrides)
    return ChatCompletionRequest(**base)


@respx.mock
async def test_happy_path_dialect_and_parsing():
    route = respx.post(f"{BASE}/gemini-2.0-flash:generateContent").respond(200, json=FIXTURE)
    response = await adapter().complete(
        req(max_tokens=100, top_p=0.9, stop="END")
    )

    # ── the sent side: the dialect contract ──
    sent = json.loads(route.calls.last.request.content)
    assert "model" not in sent                       # model lives in the URL, not the body
    assert sent["systemInstruction"] == {"parts": [{"text": "Be terse."}]}
    assert [c["role"] for c in sent["contents"]] == ["user", "model", "user"]  # assistant → model
    assert sent["contents"][0]["parts"] == [{"text": "hi"}]
    assert sent["generationConfig"]["maxOutputTokens"] == 100
    assert sent["generationConfig"]["topP"] == 0.9
    assert sent["generationConfig"]["stopSequences"] == ["END"]  # bare string → wrapped
    assert route.calls.last.request.headers["x-goog-api-key"] == "test-key"

    # ── the returned side: normalized to OpenAI schema ──
    assert response.choices[0].message.content == "Hi there! How can I help?"  # parts joined
    assert response.choices[0].finish_reason == "stop"      # STOP → stop
    assert response.usage.prompt_tokens == 8                # promptTokenCount → prompt_tokens
    assert response.usage.total_tokens == 14
    assert response.model == "gemini-2.0-flash"


@respx.mock
async def test_real_model_name_passes_through_in_the_url():
    route = respx.post(f"{BASE}/gemini-1.5-pro:generateContent").respond(200, json=FIXTURE)
    await adapter().complete(req(model="gemini-1.5-pro"))
    assert "gemini-1.5-pro:generateContent" in str(route.calls.last.request.url)


@respx.mock
async def test_400_invalid_key_is_an_embedded_rejection():
    respx.post(f"{BASE}/gemini-2.0-flash:generateContent").respond(
        400,
        json={"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                        "status": "INVALID_ARGUMENT"}},
    )
    with pytest.raises(ProviderRejected) as exc_info:
        await adapter().complete(req())
    assert "API key not valid" in str(exc_info.value)
    assert exc_info.value.retryable is False


@respx.mock
async def test_429_rate_limit_is_retryable():
    # the free tier will produce these for real — the retry layer eats them
    respx.post(f"{BASE}/gemini-2.0-flash:generateContent").respond(
        429,
        json={"error": {"code": 429, "message": "Resource has been exhausted.",
                        "status": "RESOURCE_EXHAUSTED"}},
    )
    with pytest.raises(ProviderFailure) as exc_info:
        await adapter().complete(req())
    assert exc_info.value.retryable is True


@respx.mock
async def test_500_is_a_retryable_failure():
    respx.post(f"{BASE}/gemini-2.0-flash:generateContent").respond(
        500, json={"error": {"code": 500, "message": "internal", "status": "INTERNAL"}}
    )
    with pytest.raises(ProviderFailure) as exc_info:
        await adapter().complete(req())
    assert exc_info.value.retryable is True


@respx.mock
async def test_safety_block_with_no_candidates_is_a_rejection():
    respx.post(f"{BASE}/gemini-2.0-flash:generateContent").respond(
        200, json={"promptFeedback": {"blockReason": "SAFETY"}}
    )
    with pytest.raises(ProviderRejected, match="SAFETY"):
        await adapter().complete(req())  # retrying the same prompt is pointless


@respx.mock
async def test_failure_messages_name_the_provider():
    respx.post(f"{BASE}/gemini-2.0-flash:generateContent").respond(
        500, json={"error": {"code": 500, "message": "melted"}}
    )
    with pytest.raises(ProviderFailure, match="^gemini "):
        await adapter().complete(req())