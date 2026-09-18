"""All provider-dialect translation lives here (DESIGN.md §5.1).

Adapters are thin transport; every mapping decision — system extraction,
``max_tokens`` defaulting, ``stop`` → ``stop_sequences``, response remapping —
is in this module, unit-testable with plain dicts and zero HTTP. The parameter
whitelist is enforced here: only these fields ever leave the gateway.
"""

from __future__ import annotations

import time
import uuid
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


# Gemini finishReason → OpenAI finish_reason
_GEMINI_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
}


def build_gemini_body(request: ChatCompletionRequest) -> dict[str, Any]:
    """OpenAI → Gemini generateContent. The mappings:

    - system message(s) → top-level ``systemInstruction`` (Gemini rejects the
      system role inside ``contents``); joined like the Anthropic system
    - role "assistant" → "model" — Gemini's name for the same thing
    - every message's string content → ``parts: [{text}]``
    - sampling params collapse into ``generationConfig``: temperature,
      top_p → topP, max_tokens → maxOutputTokens (OPTIONAL in Gemini — no
      defaulting needed, unlike Anthropic), stop → stopSequences (wrapped to
      a list if a bare string)
    - the model is NOT in the body — it lives in the URL path
      (…/models/{model}:generateContent). The adapter owns that.
    """
    system_parts = [m.content for m in request.messages if m.role == "system"]
    contents = [
        {"role": "model" if m.role == "assistant" else m.role, "parts": [{"text": m.content}]}
        for m in request.messages
        if m.role != "system"
    ]
    if not contents:
        raise ProviderRejected(
            "gemini requires at least one non-system message "
            "(a system-only conversation has no completion to generate)"
        )
    body: dict[str, Any] = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    generation_config: dict[str, Any] = {}
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if request.top_p is not None:
        generation_config["topP"] = request.top_p
    if request.max_tokens is not None:
        generation_config["maxOutputTokens"] = request.max_tokens
    if request.stop is not None:
        stop = request.stop if isinstance(request.stop, list) else [request.stop]
        generation_config["stopSequences"] = stop
    if generation_config:
        body["generationConfig"] = generation_config
    return body


def parse_gemini_response(payload: dict[str, Any], serving_model: str) -> ChatCompletionResponse:
    """Gemini → OpenAI. candidates[0] becomes the single choice; text parts
    are joined (same rule as the Anthropic parser); ``usageMetadata`` tokens
    are renamed. Gemini sends no timestamp and no response id — both
    synthesized at parse time.

    A response with NO candidates (prompt-level safety block) is a rejection,
    not a response: retrying is pointless (same prompt, same block), but the
    fallback chain may still serve it — exactly what a chain is for.
    Candidate-level SAFETY finish reasons map to OpenAI's ``content_filter``
    and are served as-is."""
    candidates = payload.get("candidates") or []
    if not candidates:
        block = (payload.get("promptFeedback") or {}).get("blockReason", "unknown")
        raise ProviderRejected(f"gemini blocked the prompt (blockReason={block})")

    candidate = candidates[0]
    text = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", []))
    usage_meta = payload.get("usageMetadata", {})
    in_tok = usage_meta.get("promptTokenCount", 0)
    out_tok = usage_meta.get("candidatesTokenCount", 0)
    return ChatCompletionResponse(
        id=f"chatcmpl-gemini-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=payload.get("modelVersion", serving_model),
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason=_GEMINI_FINISH_REASONS.get(
                    candidate.get("finishReason", "STOP"), "stop"
                ),
            )
        ],
        usage=Usage(prompt_tokens=in_tok, completion_tokens=out_tok, total_tokens=in_tok + out_tok),
    )
