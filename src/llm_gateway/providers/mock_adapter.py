"""MockProvider — a first-class adapter, not a stub (ADR-0006).

Same interface as the real providers, configurable latency and error rate.
Its output is *deterministic*: ids and ``created`` derive from the prompt
hash, so identical requests produce byte-identical responses. That determinism
is what lets Day 3's cache-correctness tests assert "same prompt → same
output" and "similar prompt → must NOT get that output."
"""

from __future__ import annotations

import asyncio
import hashlib
import random

from llm_gateway.config import MockSettings
from llm_gateway.providers.base import ProviderAdapter
from llm_gateway.schemas.errors import ProviderFailure
from llm_gateway.schemas.openai_api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

# Real providers stamp time.time(); the mock derives a pseudo-timestamp from
# the prompt hash so responses are byte-identical across calls.
_DETERMINISTIC_EPOCH_BASE = 1_700_000_000


class MockProvider(ProviderAdapter):
    name = "mock"

    def __init__(self, settings: MockSettings, *, name: str = "mock") -> None:
        self.name = name  # instance attr — lets one MockProvider class play many roles
        self._settings = settings

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        # Simulated network/model latency — slow enough that cache wins are
        # visible in the Day-3+ demos (default 800ms).
        await asyncio.sleep(self._settings.latency_ms / 1000.0)

        if self._settings.error_rate and random.random() < self._settings.error_rate:
            raise ProviderFailure(
                f"mock: simulated failure (error_rate={self._settings.error_rate})"
            )

        prompt = request.last_user_message() or ""
        digest = hashlib.sha256(f"{request.model}|{prompt}".encode()).hexdigest()
        content = f"[MOCK] Deterministic reply to {prompt!r} — served by the mock adapter."
        tokens_in = max(1, len(prompt.split()))
        tokens_out = max(1, len(content.split()))

        return ChatCompletionResponse(
            id=f"chatcmpl-mock-{digest[:24]}",
            created=_DETERMINISTIC_EPOCH_BASE + int(digest[:8], 16) % 86_400,
            model="mock",  # the serving model, not what the client asked for
            choices=[Choice(index=0, message=Message(role="assistant", content=content))],
            usage=Usage(
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                total_tokens=tokens_in + tokens_out,
            ),
        )

    async def health_check(self) -> bool:
        # Day 6's chaos script wants a kill switch here (force-unhealthy);
        # noted in ADR-0006's future-work list, not built today.
        return True
