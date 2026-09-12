"""All provider-dialect translation lives here (DESIGN.md §5.1).

Adapters are thin transport; every mapping decision — system extraction,
``max_tokens`` defaulting, ``stop`` → ``stop_sequences``, response remapping —
is in this module, unit-testable with plain dicts and zero HTTP. The parameter
whitelist is enforced here: only these fields ever leave the gateway.
"""

from __future__ import annotations

import time
from typing import Any, NoReturn

from llm_gateway.schemas.errors import ProviderFailure, ProviderRejected
from llm_gateway.schemas.openai_api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

# Anthropic stop_reason → OpenAI finish_reason
_STOP_REASON_MAP = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}


# ── request building ──────────────────────────────────────────────────────


def build_openai_body(request: ChatCompletionRequest, serving_model: str) -> dict[str, Any]:
    """Near-identity. None params are OMITTED, never sent as nulls — sending
    nulls is how you discover providers validate differently."""
    body: dict[str, Any] = {
        "model": serving_model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop is not None:
        body["stop"] = request.stop
    return body


def build_anthropic_body(request: ChatCompletionRequest, serving_model: str) -> dict[str, Any]:
    """OpenAI → Anthropic. The mappings that bite, each deliberate:

    - system message(s) → top-level ``system`` (Anthropic rejects the role
      inside ``messages``); multiple system messages are joined with blank lines
    - ``max_tokens`` is REQUIRED by Anthropic but optional in OpenAI → default 1024
    - ``stop`` → ``stop_sequences``
    """
    system_parts = [m.content for m in request.messages if m.role == "system"]
    messages = [m for m in request.messages if m.role != "system"]
    if not messages:
        raise ProviderRejected(
            "anthropic requires at least one non-system message "
            "(a system-only conversation has no completion to generate)"
        )
    body: dict[str, Any] = {
        "model": serving_model,
        "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop is not None:
        body["stop_sequences"] = request.stop
    return body


# ── response parsing ──────────────────────────────────────────────────────


def parse_openai_response(payload: dict[str, Any]) -> ChatCompletionResponse:
    """OpenAI→OpenAI is identity in content — but parse, don't trust: the
    contract 'output is always the OpenAI schema' is enforced by parsing."""
    return ChatCompletionResponse.model_validate(payload)


def parse_anthropic_response(payload: dict[str, Any], serving_model: str) -> ChatCompletionResponse:
    """Anthropic → OpenAI: join text blocks, remap stop_reason, rename usage.
    Anthropic sends no timestamp, so ``created`` is stamped at parse time."""
    text = "".join(
        block["text"] for block in payload.get("content", []) if block.get("type") == "text"
    )
    in_tok = payload.get("usage", {}).get("input_tokens", 0)
    out_tok = payload.get("usage", {}).get("output_tokens", 0)
    return ChatCompletionResponse(
        id=payload.get("id", "chatcmpl-unknown"),
        created=int(time.time()),
        model=payload.get("model", serving_model),
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason=_STOP_REASON_MAP.get(payload.get("stop_reason", "end_turn"), "stop"),
            )
        ],
        usage=Usage(prompt_tokens=in_tok, completion_tokens=out_tok, total_tokens=in_tok + out_tok),
    )


# ── error classification ──────────────────────────────────────────────────


def classify_provider_error(*, status_code: int, provider: str, body: str) -> NoReturn:
    """HTTP status → the right exception. 4xx (except 429) = rejected, not
    retryable; 429 and 5xx = transient, retryable. Network-level failures
    never reach here — adapters raise ProviderFailure for those directly."""
    if 400 <= status_code < 500 and status_code != 429:
        raise ProviderRejected(
            f"{provider} rejected the request (HTTP {status_code}): {body[:500]}"
        )
    raise ProviderFailure(f"{provider} failed (HTTP {status_code}): {body[:500]}")
