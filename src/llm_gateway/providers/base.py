"""The provider seam: one protocol, one registry (DESIGN.md §5.4).

Contract: ``complete()`` ALWAYS returns an OpenAI-schema ``ChatCompletionResponse``.
Normalization is the adapter's job (Day 2's ``normalizer.py`` does it for the
real providers), never the caller's.

The registry is built per-app — a factory, not a module-level global — so
tests can construct an isolated app (e.g. a mock with ``error_rate=1.0``)
without polluting anything else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_gateway.config import Settings
from llm_gateway.schemas.openai_api import ChatCompletionRequest, ChatCompletionResponse


class ProviderAdapter(ABC):
    """Every provider — mock, OpenAI, Anthropic, future local — implements this."""

    name: str

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Serve one chat completion. Raises GatewayError subclasses on failure."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Used by /v1/health and (Day 6) the chaos tooling."""


class ProviderRegistry:
    """Mutable registry + ordered default chain. One instance per app."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderAdapter] = {}
        self._chain: list[str] = []

    def register(self, adapter: ProviderAdapter, *, in_default_chain: bool = True) -> None:
        if adapter.name in self._providers:
            raise ValueError(f"duplicate provider name: {adapter.name!r}")
        self._providers[adapter.name] = adapter
        if in_default_chain:
            self._chain.append(adapter.name)

    def get(self, name: str) -> ProviderAdapter:
        try:
            return self._providers[name]
        except KeyError:
            raise LookupError(
                f"unknown provider {name!r}; registered: {sorted(self._providers)}"
            ) from None

    def default_chain(self) -> list[ProviderAdapter]:
        return [self._providers[name] for name in self._chain]

    def all(self) -> list[ProviderAdapter]:
        return list(self._providers.values())

    def __len__(self) -> int:
        return len(self._providers)


def build_registry(settings: Settings) -> ProviderRegistry:
    """Assemble the registry from settings.

    The mock is always available (ADR-0006). Day 2 registers real adapters
    ahead of it when their API keys are set — the mock stays the terminator.
    """
    from llm_gateway.providers.mock_adapter import MockProvider  # breaks the import cycle

    registry = ProviderRegistry()
    if settings.mock.enabled:
        registry.register(MockProvider(settings.mock), in_default_chain=True)
    if not registry.default_chain():
        raise RuntimeError(
            "no providers registered — enable the mock (GW_MOCK__ENABLED) or set provider keys"
        )
    return registry
