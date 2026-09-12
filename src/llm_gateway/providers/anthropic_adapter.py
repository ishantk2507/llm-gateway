"""Anthropic adapter — transport only. The three dialect differences (auth
headers, /v1/messages, system-as-top-level-field) all live in the normalizer;
this file just carries bytes."""

from __future__ import annotations

import httpx

from llm_gateway.config import AnthropicProviderSettings
from llm_gateway.providers.base import ProviderAdapter
from llm_gateway.providers.normalizer import (
    build_anthropic_body,
    classify_provider_error,
    parse_anthropic_response,
)
from llm_gateway.schemas.errors import ProviderFailure
from llm_gateway.schemas.openai_api import (
    VIRTUAL_MODELS,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"

    def __init__(self, settings: AnthropicProviderSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={
                "x-api-key": settings.api_key.get_secret_value(),
                "anthropic-version": "2023-06-01",
            },
            timeout=settings.timeout_s,
        )

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        serving_model = self._settings.model if request.model in VIRTUAL_MODELS else request.model
        body = build_anthropic_body(request, serving_model)
        try:
            response = await self._client.post("/v1/messages", json=body)
        except httpx.HTTPError as exc:
            raise ProviderFailure(f"anthropic: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            classify_provider_error(
                status_code=response.status_code, provider="anthropic", body=response.text
            )
        return parse_anthropic_response(response.json(), serving_model)

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/v1/models")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()
