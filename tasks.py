"""Task runner for llm-gateway -- one entry point on every OS.

    python tasks.py <target> [extra args...]     (any OS)
    make <target>                                 (Windows: via make.bat)

Targets that accept extra args pass them to the underlying tool, e.g.:

    python tasks.py test -k test_health
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sh(*cmd: str) -> None:
    """Run a command from the project root. List-args, no shell involved,
    so quoting behaves identically on Windows, macOS, and Linux."""
    print(f"> {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from None


# ── targets ──────────────────────────────────────────────────────────────


def install() -> None:
    sh("uv", "sync")


def run(*args: str) -> None:
    sh(
        "uv",
        "run",
        "uvicorn",
        "llm_gateway.main:create_app",
        "--factory",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        *args,
    )


def test(*args: str) -> None:
    sh("uv", "run", "pytest", *args)


def cov(*args: str) -> None:
    sh("uv", "run", "pytest", "--cov=llm_gateway", "--cov-report=term-missing", *args)


def lint(*args: str) -> None:
    sh("uv", "run", "ruff", "check", ".", *args)


def fmt() -> None:
    sh("uv", "run", "ruff", "check", "--fix", ".")
    sh("uv", "run", "ruff", "format", ".")


def lock() -> None:
    sh("uv", "lock")


def compose_up() -> None:
    sh("docker", "compose", "up", "-d")


def compose_down() -> None:
    sh("docker", "compose", "down")


def logs() -> None:
    sh("docker", "compose", "logs", "-f", "gateway")


def eval() -> None:
    sh("uv", "run", "python", "evals/run_router_eval.py")


def clean() -> None:
    for name in (".pytest_cache", ".ruff_cache", "htmlcov"):
        shutil.rmtree(ROOT / name, ignore_errors=True)
    (ROOT / ".coverage").unlink(missing_ok=True)
    for cache in ROOT.rglob("__pycache__"):
        if ".venv" not in cache.parts:  # never touch the venv
            shutil.rmtree(cache, ignore_errors=True)


# Placeholders so the muscle memory exists from day one; each fills in
# when its phase lands.


def _not_yet(day: str) -> Callable[[], None]:
    def target() -> None:
        print(f"[placeholder] this target lands on {day}.")
        raise SystemExit(1)

    return target


TARGETS: dict[str, tuple[str, Callable]] = {
    "install": ("uv sync -- dependencies exactly as locked", install),
    "run": ("uvicorn with auto-reload on :8000", run),
    "test": ("run the full test suite", test),
    "cov": ("tests with coverage report", cov),
    "lint": ("ruff check (CI runs exactly this)", lint),
    "fmt": ("ruff autofix + format (run before pushing)", fmt),
    "lock": ("regenerate uv.lock -- commit the result", lock),
    "compose-up": ("full stack: gateway+redis+postgres+prometheus+grafana", compose_up),
    "compose-down": ("stop the stack (volumes kept)", compose_down),
    "logs": ("tail gateway logs", logs),
    "clean": ("delete caches and artifacts (never touches .venv)", clean),
    "eval": ("router golden-set eval", eval),
    "load-test": ("locust load test", _not_yet("Day 6")),
    "chaos": ("kill primary provider under load", _not_yet("Day 6")),
    "demo": ("run the 5-minute demo", _not_yet("Day 7")),
}

# Targets allowed extra CLI args, e.g. `make test -k health`
PASSTHROUGH = {"run", "test", "cov", "lint"}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in {"help", "-h", "--help"}:
        print(__doc__)
        print("targets:")
        width = max(len(name) for name in TARGETS)
        for name, (desc, _) in TARGETS.items():
            print(f"  {name:<{width}}  {desc}")
        return

    name, extra = args[0], args[1:]
    if name not in TARGETS:
        print(f"unknown target: {name!r} -- try `make help`")
        raise SystemExit(2)
    if extra and name not in PASSTHROUGH:
        print(f"target {name!r} takes no extra args (got: {extra})")
        raise SystemExit(2)

    desc, fn = TARGETS[name]
    print(f"==> {name}: {desc}")
    fn(*extra)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130) from None
