"""OpenAI Chat Completions-compatible request/response models (DESIGN.md §5.1, §7).

Two deliberate deviations from "just mirror the spec":

- ``extra="allow"`` on the models: unknown parameters are *dropped with a
  logged warning* (the route calls ``dropped_parameters()``), never silently
  forwarded — silent parameter drift across providers is a correctness bug.
- ``stream`` is parsed but not validated here: a validator would surface as a
  FastAPI 422, and the v1 contract is a proper 400 envelope. The route
  raises ``StreamNotSupported`` after parsing succeeds.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
VIRTUAL_MODELS: frozenset[str] = frozenset({"auto", "cheap", "standard", "premium", "mock"})


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: str = Field(
        ...,
        description="v1 is text-only; multimodal content arrays are rejected with a 400",
    )


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(
        ...,
        description="virtual model: 'auto' | tier name | real model name | 'mock' (§5.1)",
    )
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    stream: bool = False

    def dropped_parameters(self) -> list[str]:
        """Unknown top-level parameters, sorted — logged by the route, never forwarded."""
        extras = getattr(self, "__pydantic_extra__", None) or {}
        return sorted(extras)

    def last_user_message(self) -> str | None:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return None

    def is_single_turn(self) -> bool:
        """Exactly one user message, optionally preceded by one system message.

        The cacheability gate for Day 3 (ADR-0003) — defined now because it
        is a property of the request shape, not of the cache.
        """
        roles = [m.role for m in self.messages]
        return (
            roles.count("user") == 1
            and roles.count("system") <= 1
            and all(role in {"system", "user"} for role in roles)
        )


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str  # the *serving* model, not what the client asked for
    choices: list[Choice] = Field(min_length=1)
    usage: Usage
