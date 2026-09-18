"""Gemini adapter — transport only; all translation lives in normalizer.py.

One attempt, like every adapter (retry/fallback live above). The dialect
quirk this adapter owns: the model lives in the URL path, not the body —
so virtual-model substitution happens here, at URL-build time."""

from __future__ import annotations

import httpx

from llm_gateway.config import GeminiProviderSettings
from llm_gateway.providers.base import ProviderAdapter
from llm_gateway.providers.normalizer import (
    build_gemini_body,
    classify_provider_error,
    parse_gemini_response,
)
from llm_gateway.schemas.errors import ProviderFailure
from llm_gateway.schemas.openai_api import (
    VIRTUAL_MODELS,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

_API_VERSION = "v1beta"


class GeminiAdapter(ProviderAdapter):
    name = "gemini"

    def __init__(self, settings: GeminiProviderSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"x-goog-api-key": settings.api_key.get_secret_value()},
            timeout=settings.timeout_s,
        )

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        serving_model = self._settings.model if request.model in VIRTUAL_MODELS else request.model
        body = build_gemini_body(request)
        url = f"/{_API_VERSION}/models/{serving_model}:generateContent"
        try:
            response = await self._client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise ProviderFailure(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            classify_provider_error(
                status_code=response.status_code, provider=self.name, body=response.text
            )
        return parse_gemini_response(response.json(), serving_model)

    async def health_check(self) -> bool:
        """GET /v1beta/models — one round trip that genuinely answers
        'key valid + reachable'."""
        try:
            response = await self._client.get(f"/{_API_VERSION}/models")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()
