"""Cacheability policy — 'may we cache this?' (DESIGN.md §5.2, ADR-0003).

Pure functions, no I/O: the whole module is unit-testable without Redis,
FAISS, or a model. Single source of truth for cache identity — the hash is
shared by BOTH layers, so invalidation by prompt_hash addresses each.
"""

from __future__ import annotations

import hashlib
import re

from llm_gateway.schemas.openai_api import VIRTUAL_MODELS, ChatCompletionRequest

MAX_CACHED_TEMPERATURE = 0.3
_TOOL_PARAM_NAMES = frozenset({"tools", "tool_choice", "functions", "function_call"})
_WHITESPACE = re.compile(r"\s+")


def is_cacheable(request: ChatCompletionRequest) -> bool:
    """All gates must hold. Multi-turn is the #1 correctness hazard (ADR-0003):
    follow-ups near-match the tails of unrelated conversations."""
    if request.stream:
        return False
    if request.temperature is not None and request.temperature > MAX_CACHED_TEMPERATURE:
        return False
    extras = getattr(request, "__pydantic_extra__", None) or {}
    if _TOOL_PARAM_NAMES & extras.keys():
        return False
    return request.is_single_turn()


def normalized_prompt(request: ChatCompletionRequest) -> str:
    """The cache identity of the prompt: last user message, whitespace-collapsed.
    Precondition: is_cacheable(request) — which guarantees a user message exists."""
    content = request.last_user_message() or ""
    return _WHITESPACE.sub(" ", content).strip()


def model_family(model: str) -> str:
    """Which family a request belongs to — derivable BEFORE routing, because
    the cache is consulted before the router is."""
    if model == "mock":
        return "mock"
    if model in VIRTUAL_MODELS:
        return "routed"
    if model.startswith(("gpt-", "o1", "o3", "chatgpt")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    return "other"


def temperature_bucket(request: ChatCompletionRequest) -> str:
    """Only 'low' is ever cached (§5.2); the bucket exists in the key so a
    future 'high' bucket doesn't require rehashing the world."""
    return "low"


def prompt_hash(normalized: str, family: str, bucket: str) -> str:
    return hashlib.sha256(f"{family}|{bucket}|{normalized}".encode()).hexdigest()