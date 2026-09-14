"""Cache-correctness — the adversarial spec (ADR-0002/0003).

Written BEFORE the cache existed; it dictates 'done'. Runs the REAL MiniLM
model, because only the real model can tell whether 'circuit breaker pattern'
vs 'circuit breaker home wiring' lands at ~0.87. First run downloads ~90MB.

If a paraphrase pair stops hitting at 0.95, that is not a test failure to
silence — that's threshold calibration data. Fix the pair or revisit the
threshold (benchmarks/threshold_sweep.py, Day 6).
"""

import hashlib
import time

import pytest
from fakeredis.aioredis import FakeRedis

from llm_gateway.cache.embedder import MiniLMEmbedder
from llm_gateway.cache.exact import ExactCache
from llm_gateway.cache.semantic import SemanticCache
from llm_gateway.cache.service import CacheService
from llm_gateway.config import CacheSettings
from llm_gateway.schemas.openai_api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

ADVERSARIAL_PAIRS = [
    ("Explain the circuit breaker pattern in distributed systems.",
     "Explain what a circuit breaker does in home electrical wiring."),
    ("What is the capital of France?", "What is the capital of Georgia?"),
    ("How do I restart a frozen computer?", "How do I restart a frozen Python program?"),
    ("How do I make a mint julep?", "How do I make a julep without mint?"),
    ("What is the boiling point of water?", "What is the boiling point of mercury?"),
    ("When was the first iPhone released?", "When was the first Android phone released?"),
    ("How tall is the Eiffel Tower?", "How tall is the Tokyo Tower?"),
    ("What language is spoken in Brazil?", "What language is spoken in Austria?"),
    ("What is the speed of light?", "What is the speed of sound?"),
    ("How do I tie a bowline knot?", "How do I tie a Windsor knot?"),
    ("Write a function to reverse a list in Python.",
     "Write a function to reverse a string in Python."),
    ("What are the symptoms of a cold?", "What are the symptoms of the flu?"),
]

PARAPHRASE_PAIRS = [
    ("What is the capital of France?", "What's the capital city of France?"),
    ("Explain the circuit breaker pattern in distributed systems.",
     "Can you explain how the circuit breaker pattern works in distributed systems?"),
    ("How do I reverse a list in Python?", "In Python, how can I reverse a list?"),
    ("What are the health benefits of running?", "Tell me the health benefits of running."),
]


# Discovered while calibrating: this word-order rewrite scores 0.861 with MiniLM —
# inside the adversarial band. The model cannot distinguish it from a different
# question, so it MUST near-miss; serving it would mean the threshold also serves
# wrong answers. This test pins the boundary the threshold sweep (Day 6) will chart.
EXPECTED_NEAR_MISS_PAIRS = [
    ("What are the health benefits of running?", "What does running do for your health?"),
    ("Summarize the plot of Hamlet.", "Give me a summary of the plot of Hamlet."),
]



@pytest.fixture(scope="session")
def embedder():
    return MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", threads=2)


def req(text: str, **overrides) -> ChatCompletionRequest:
    base = {"model": "auto", "messages": [Message(role="user", content=text)]}
    base.update(overrides)
    return ChatCompletionRequest(**base)


def response_for(text: str) -> ChatCompletionResponse:
    digest = hashlib.sha256(text.encode()).hexdigest()
    content = f"Cached answer for {text!r}"
    return ChatCompletionResponse(
        id=f"chatcmpl-test-{digest[:16]}",
        created=1_700_000_000,
        model="mock",
        choices=[Choice(message=Message(role="assistant", content=content))],
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


def build_service(embedder, workdir) -> CacheService:
    settings = CacheSettings()  # threshold 0.95, floor 0.85 — the defaults under test
    semantic = SemanticCache(
        embedder,
        index_path=workdir / "vectors.index",
        db_path=workdir / "payloads.db",
        settings=settings,
    )
    return CacheService(
        exact=ExactCache(FakeRedis(), ttl_seconds=3600),
        semantic=semantic,
        settings=settings,
    )


async def test_exact_hit_is_byte_identical(embedder, tmp_path):
    service = build_service(embedder, tmp_path)
    original = response_for("What is the capital of France?")
    await service.store(req("What is the capital of France?"), original)
    result = await service.lookup(req("What is the capital of France?"))
    assert result.hit and result.layer == "exact"
    assert result.response.model_dump() == original.model_dump()


async def test_paraphrases_hit_semantically(embedder, tmp_path):
    for i, (original_text, paraphrase_text) in enumerate(PARAPHRASE_PAIRS):
        service = build_service(embedder, tmp_path / f"paraphrase{i}")  # isolated per pair
        await service.store(req(original_text), response_for(original_text))
        result = await service.lookup(req(paraphrase_text))
        assert result.hit, f"paraphrase missed: {original_text!r} vs {paraphrase_text!r}"
        assert result.layer == "semantic"
        assert result.similarity >= 0.95


async def test_adversarial_pairs_never_cross_hit(embedder, tmp_path):
    """THE test: similar prompts with different correct answers must NEVER hit."""
    for i, (a, b) in enumerate(ADVERSARIAL_PAIRS):
        for direction, (stored, queried) in enumerate(((a, b), (b, a))):
            # Paths derive from indices only — prompt text can end in a space
            # ("Explain ") which Windows path handling turns into an unopenable dir.
            service = build_service(embedder, tmp_path / f"pair{i}-{direction}")
            await service.store(req(stored), response_for(stored))
            result = await service.lookup(req(queried))
            assert not result.hit, f"cross-hit: {stored!r} served for {queried!r}"

async def test_word_order_rewrites_land_in_the_near_miss_band(embedder, tmp_path):
    for i, (a, b) in enumerate(EXPECTED_NEAR_MISS_PAIRS):
        service = build_service(embedder, tmp_path / f"nm{i}")
        await service.store(req(a), response_for(a))
        result = await service.lookup(req(b))
        assert not result.hit
        assert result.near_miss
        assert 0.85 <= result.similarity < 0.95


async def test_expired_entries_miss(embedder, tmp_path):
    settings = CacheSettings(semantic_ttl_seconds=1)
    semantic = SemanticCache(
        embedder, index_path=tmp_path / "v.index", db_path=tmp_path / "p.db", settings=settings
    )
    service = CacheService(
        exact=ExactCache(FakeRedis(), ttl_seconds=1), semantic=semantic, settings=settings
    )
    await service.store(req("What is the capital of France?"), response_for("x"))
    time.sleep(1.1)
    assert not (await service.lookup(req("What is the capital of France?"))).hit


async def test_family_isolation(embedder, tmp_path):
    service = build_service(embedder, tmp_path)
    await service.store(req("What is the capital of France?", model="gpt-4o"), response_for("x"))
    result = await service.lookup(
        req("What is the capital of France?", model="claude-3-5-haiku-latest")
    )
    assert not result.hit  # openai's answer must never serve an anthropic request