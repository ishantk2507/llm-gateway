"""Guardrail behind the guardrail: the hermetic boundary must stay hermetic.

If a provider adapter is added without a pin in tests/helpers.py, this fails
loudly — instead of the suite silently making real network calls, which is
exactly what happened on the first Day-5 run (§8, hermeticity incident)."""

import os

from tests.helpers import assert_env_scrubbed, hermetic_settings

from llm_gateway.providers.base import build_registry


def test_hermetic_settings_register_mock_only():
    registry = build_registry(hermetic_settings())
    assert [p.name for p in registry.all()] == ["mock"]


def test_hermetic_pin_survives_real_keys_in_the_environment(monkeypatch):
    """The pin must beat both leak channels: exported shell vars and anything
    a future BaseSettings source invents. (Init-kwargs priority alone was
    falsified during the incident; this pins the behavior so a regression
    in either layer — conftest scrub or init kwargs — fails here first.)"""
    monkeypatch.setenv("GROQ_API_KEY", "real-looking-key")
    monkeypatch.setenv("GEMINI_API_KEY", "also-real")
    monkeypatch.setenv("OPENAI_API_KEY", "me-too")
    registry = build_registry(hermetic_settings())
    assert [p.name for p in registry.all()] == ["mock"]


def test_hostile_dotenv_in_cwd_does_not_leak(tmp_path):
    """A `.env` with real keys in the current directory must not reach the
    test app: conftest chdirs to an empty tmp_path, and the pins' own
    `_env_file=None` must defeat a group's dotenv read even if the CWD
    defense were gone."""
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=gsk-hostile\nOPENAI_API_KEY=sk-hostile\n", encoding="utf-8"
    )
    registry = build_registry(hermetic_settings())
    assert [p.name for p in registry.all()] == ["mock"]


def test_process_env_is_scrubbed():
    """conftest's autouse fixture ran before this line: no provider/gateway
    prefix may survive in the process environment — even with a real
    GROQ_API_KEY sitting in the repo's `.env` and the developer's shell."""
    assert_env_scrubbed()
    # and the CWD is the per-test tmp_path — no relative `.env` to read
    assert not os.path.exists(".env")
