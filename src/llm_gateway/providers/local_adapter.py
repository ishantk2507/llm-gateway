"""Local open-source models via Ollama (or vLLM / LM Studio / llama.cpp server).

Ollama exposes an OpenAI-compatible endpoint, so 90% of this adapter already
exists as OpenAIAdapter — transport, health check, virtual-model substitution
all inherit. The ONLY thing worth overriding is the name: without it, logs and
x-provider-used would claim 'openai' while serving Llama, and honest
observability is this project's whole identity.
"""

from __future__ import annotations

from llm_gateway.config import LocalSettings, OpenAIProviderSettings
from llm_gateway.providers.openai_adapter import OpenAIAdapter


class LocalAdapter(OpenAIAdapter):
    name = "local"

    def __init__(self, settings: LocalSettings) -> None:
        super().__init__(
            OpenAIProviderSettings(
                api_key="local",  # dummy bearer — Ollama ignores auth; transport sends the header
                base_url=settings.base_url,
                model=settings.model,
                timeout_s=settings.timeout_s,
            )
        )
