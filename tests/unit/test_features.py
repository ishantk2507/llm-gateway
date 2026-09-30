from llm_gateway.router.features import extract_features
from llm_gateway.schemas.openai_api import ChatCompletionRequest, Message


def r(text: str, history: list[Message] | None = None) -> ChatCompletionRequest:
    messages = (history or []) + [Message(role="user", content=text)]
    return ChatCompletionRequest(model="auto", messages=messages)


def test_token_count_spans_the_whole_conversation():
    history = [
        Message(role="user", content="one two three"),
        Message(role="assistant", content="four five"),
    ]
    assert extract_features(r("six", history)).token_count == 6


def test_fenced_code_detected():
    assert extract_features(r("Fix this:\n```python\nx = 1\n```")).has_code


def test_code_signals_detected_in_prose():
    assert extract_features(r("Start from `import os` and explain.")).has_code


def test_prose_does_not_trigger_code_detection():
    # The false-positive traps: "class" in sociology, "function of the liver"
    assert not extract_features(r("What is social class in sociology?")).has_code
    assert not extract_features(r("What is the function of the liver?")).has_code


def test_structured_data_detected():
    assert extract_features(r('Parse this: {"a": 1}')).has_structured_data
    assert not extract_features(r("Just a normal sentence.")).has_structured_data


def test_keyword_hits_respect_word_boundaries():
    features = extract_features(r("What is the history of Rome?"), keywords=("hi", "prove"))
    assert features.keyword_hits == ()  # 'hi' must NOT match inside 'history'
