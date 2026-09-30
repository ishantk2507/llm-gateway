"""THE hermetic boundary (ADR-0006) — enforced at the process level.

The suite must be green WITH real credentials sitting in `.env` (a gateway
that can only pass its tests with its production config deleted is lying to
itself). Two channels let the outside world into a test process:

1. `.env` — every provider BaseSettings group reads its OWN env_file, resolved
   relative to the CWD, and the parent's `_env_file=None` does not propagate
   to them. Chdir to a per-test tmp_path: the relative ".env" then resolves
   to a file that does not exist.
2. exported shell vars — GROQ_API_KEY etc. in the developer's shell. Scrub
   every SCRUB_PREFIXES variable from the process environment.

Both closed HERE, at the process level, where pydantic source priorities are
irrelevant — the init-kwarg pins in tests/helpers.py stay as defense-in-depth
for anything that re-reads sources after this fixture has run (and for the
documented case that init-kwargs do not reliably outrank env/dotenv; the
first Day-5 run falsified that assumption empirically).
"""

import os

import pytest

from tests.helpers import SCRUB_PREFIXES


@pytest.fixture(autouse=True)
def hermetic_process(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Every test runs from an empty CWD with provider env vars scrubbed."""
    for name in [k for k in os.environ if k.startswith(SCRUB_PREFIXES)]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield
