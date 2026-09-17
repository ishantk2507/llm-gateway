"""Pure feature extraction — request in, numbers out, no I/O (DESIGN.md §5.3).

token_count is a whitespace approximation of tokens — good enough for coarse
rule bounds, and documented as such. Code detection errs toward PRECISION over
recall: false positives route easy prompts to premium (wasted money), false
negatives just get a standard answer (fine). The keyword lists in
data/keywords.yaml carry the recall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llm_gateway.schemas.openai_api import ChatCompletionRequest

# Conservative signals — each one is near-impossible in prose.
CODE_SIGNALS = ("```", "def ", "import ", "console.log", "#include", "func ", "=>", "();")
# JSON-ish structures pasted into prompts.
STRUCTURED_SIGNALS = ('{"', '"}', '":', "[{")


@dataclass(frozen=True)
class PromptFeatures:
    token_count: int
    has_code: bool
    has_structured_data: bool
    keyword_hits: tuple[str, ...]


def extract_features(
    request: ChatCompletionRequest, keywords: tuple[str, ...] = ()
) -> PromptFeatures:
    """Tokens count over the WHOLE conversation (coarse length signal); code,
    structure, and keyword detection look at the LAST user message — the
    current ask is what determines difficulty, history is context."""
    all_text = " ".join(m.content for m in request.messages)
    ask = (request.last_user_message() or "").lower()

    hits = tuple(k for k in keywords if re.search(rf"\b{re.escape(k)}\b", ask))
    return PromptFeatures(
        token_count=len(all_text.split()),
        has_code=any(signal in ask for signal in CODE_SIGNALS),
        has_structured_data=any(signal in ask for signal in STRUCTURED_SIGNALS),
        keyword_hits=hits,
    )
