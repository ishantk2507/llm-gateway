"""OpenAI adapter — transport only; all translation lives in normalizer.py.

Makes exactly ONE attempt. Retrying and falling back are layers above
(reliability/), which is what keeps this contract-testable.
"""

from __future__ import annotations

import httpx

from llm_gateway.config import OpenAIProviderSettings
from llm_gateway.providers.base import ProviderAdapter
from llm_gateway.providers.normalizer import (
    build_openai_body,
    classify_provider_error,
    parse_openai_response,
)
from llm_gateway.schemas.errors import ProviderFailure
from llm_gateway.schemas.openai_api import (
    VIRTUAL_MODELS,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class OpenAIAdapter(ProviderAdapter):
    name = "openai"

    def __init__(self, settings: OpenAIProviderSettings) -> None:
        self._settings = settings
        # One client for the adapter's lifetime (connection reuse). Its
        # timeout is a lower bound of defense — retry.py's wait_for slices
        # the real budget above it.
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
            timeout=settings.timeout_s,
        )

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        serving_model = self._settings.model if request.model in VIRTUAL_MODELS else request.model
        body = build_openai_body(request, serving_model)
        try:
            response = await self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:  # connect, read, timeout, DNS — transient by assumption
            raise ProviderFailure(f"openai: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            classify_provider_error(
                status_code=response.status_code, provider="openai", body=response.text
            )
        return parse_openai_response(response.json())

    async def health_check(self) -> bool:
        """One cheap round trip that genuinely answers 'key valid + reachable'."""
        try:
            response = await self._client.get("/models")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()
