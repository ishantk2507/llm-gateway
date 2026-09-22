from pathlib import Path

from llm_gateway.config import Tier
from llm_gateway.router.classifier import RuleBasedClassifier

DATA = Path(__file__).resolve().parents[2] / "data" / "keywords.yaml"


def classify(text: str):
    from llm_gateway.schemas.openai_api import ChatCompletionRequest, Message

    return RuleBasedClassifier.from_yaml(DATA).classify(
        ChatCompletionRequest(model="auto", messages=[Message(role="user", content=text)])
    )


def test_greeting_routes_cheap():
    d = classify("Hello, gateway.")
    assert d.tier is Tier.CHEAP and d.rule_fired == "greeting"


def test_code_routes_premium():
    d = classify("Write a Python function that reverses a linked list.")
    assert d.tier is Tier.PREMIUM and d.rule_fired == "write_code_with_language"


def test_reasoning_routes_premium():
    assert classify("Prove that the square root of 2 is irrational.").rule_fired == "deep_reasoning"


def test_explain_routes_standard():
    d = classify("Explain the circuit breaker pattern in distributed systems.")
    assert d.tier is Tier.STANDARD and d.rule_fired == "explain_concept"


def test_short_factual_routes_cheap():
    assert classify("What is the capital of France?").rule_fired == "simple_factual"


def test_domain_guard_blocks_simple_factual_for_concepts():
    # "What is OAuth?" is 3 tokens and starts with "what is" — the DOMAIN
    # guard is what keeps it out of cheap (rubric edge rule 6).
    d = classify("What is OAuth?")
    assert d.tier is Tier.STANDARD


def test_explain_gravity_stays_standard():
    d = classify("Explain gravity.")
    assert d.tier is Tier.STANDARD


def test_no_match_falls_to_default():
    d = classify("Tell me something interesting about octopuses.")
    assert d.tier is Tier.STANDARD and d.rule_fired == "default"


def test_first_match_wins_code_over_reasoning():
    # matches code_task (listed first) even though 'design' also implies premium
    d = classify("Implement a function, then design a test plan for it.")
    assert d.rule_fired == "code_task"
