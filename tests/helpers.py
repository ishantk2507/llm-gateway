"""Shared test plumbing — defense-in-depth half of the hermetic boundary (ADR-0006).

THE boundary itself is tests/conftest.py (per-test CWD + env scrub, enforced
at the process level). The pins here are the second layer: nested provider
BaseSettings groups (OpenAI/Gemini/Groq/…) read their OWN sources when
instantiated — env vars and `.env` via their own model_config — and the parent
Settings' `_env_file=None` does NOT propagate to them. On the first Day-5
test run a real GROQ_API_KEY sitting correctly in `.env` leaked in, the suite
registered a live adapter, and made real, billed network calls.

Do not trust init-kwargs to outrank env/dotenv: that assumption was falsified
empirically during that incident (and `Settings` has `extra="ignore"`, which
silently drops a mistyped group kwarg — the original `groq=` pin never
applied because the field is named `grok`). The pins are still worth keeping:
they hold the invariant even if a future test re-reads sources or a conftest
refactor loosens the process-level scrub.

Adding a new adapter? Add its pin here — or the guardrail test
(tests/unit/test_hermetic.py) fails before any test touches the network.
"""

import os

from pydantic import SecretStr

from llm_gateway.config import (
    AnthropicProviderSettings,
    AuthSettings,
    GeminiProviderSettings,
    GroqProviderSettings,
    LocalSettings,
    OpenAIProviderSettings,
    Settings,
)

# Every environment variable prefix that can inject provider credentials or
# gateway config into a test process. conftest.py deletes all of these;
# this list is the single source of truth for both files.
SCRUB_PREFIXES = ("GW_", "OPENAI_", "ANTHROPIC_", "GEMINI_", "GROQ_")


def hermetic_settings(**groups) -> Settings:
    """Settings with every provider pinned off — no .env, no env vars, no network.

    All integration-test `make_settings()` helpers delegate here. Test files
    override `app`, `mock`, `reliability`, `cache` freely; provider pins are
    the default, not an option.
    """
    pinned = dict(
        openai=OpenAIProviderSettings(api_key=SecretStr(""), _env_file=None),
        anthropic=AnthropicProviderSettings(api_key=SecretStr(""), _env_file=None),
        gemini=GeminiProviderSettings(api_key=SecretStr(""), _env_file=None),
        grok=GroqProviderSettings(api_key=SecretStr(""), _env_file=None),
        local=LocalSettings(enabled=False),
        auth=AuthSettings(api_keys=set()),  # open mode — the suite never sends headers
    )
    pinned.update(groups)
    return Settings(_env_file=None, **pinned)


def assert_env_scrubbed() -> None:
    """Fail if any scrubbed prefix survived into this test's process env."""
    leaked = [k for k in os.environ if k.startswith(SCRUB_PREFIXES)]
    assert not leaked, f"env scrub failed, still set: {leaked}"
