"""Groq adapter — OpenAI-compatible transport (inherits OpenAIAdapter);
the ONLY override is the name: logs and x-provider-used must say 'groq',
not 'openai', because honest observability is this project's identity."""

from __future__ import annotations

from llm_gateway.config import GroqProviderSettings, OpenAIProviderSettings
from llm_gateway.providers.openai_adapter import OpenAIAdapter


class GroqAdapter(OpenAIAdapter):
    name = "groq"

    def __init__(self, settings: GroqProviderSettings) -> None:
        super().__init__(
            OpenAIProviderSettings(
                api_key=settings.api_key,
                base_url=settings.base_url,
                model=settings.model,
                timeout_s=settings.timeout_s,
            )
        )
