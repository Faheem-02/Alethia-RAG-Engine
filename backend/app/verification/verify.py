"""Verification module.

Responsibilities:
- Validate that generated output is supported by retrieved evidence.
- Return verification signals for downstream response handling.
- Keep policy checks separate from generation logic.
"""

import re
from typing import Any, Dict, List

from backend.app.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens."""
    return re.findall(r"\w+", text.lower())


def _overlap_ratio(answer: str, context_text: str) -> float:
    """Compute lexical overlap ratio between answer and a context chunk."""
    answer_terms = set(_tokenize(answer))
    if not answer_terms:
        return 0.0

    context_terms = set(_tokenize(context_text))
    overlap_count = len(answer_terms.intersection(context_terms))
    return overlap_count / len(answer_terms)


def verify_answer(answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify answer support via simple lexical overlap with context."""
    logger.info("Verifying answer against %d context chunk(s).", len(context_chunks))

    if not answer.strip() or not context_chunks:
        return {"answer": answer, "confidence": 0.0, "sources": []}

    source_scores: Dict[str, float] = {}
    for chunk in context_chunks:
        context_text = str(chunk.get("text", ""))
        score = _overlap_ratio(answer=answer, context_text=context_text)
        source = str(chunk.get("source", "unknown"))
        if score > source_scores.get(source, 0.0):
            source_scores[source] = score

    # Confidence is the best overlap score clipped to [0, 1].
    confidence = max(source_scores.values(), default=0.0)
    confidence = max(0.0, min(1.0, float(confidence)))

    ranked_sources = sorted(source_scores.items(), key=lambda item: item[1], reverse=True)
    sources = [source for source, score in ranked_sources if score > 0.0]

    return {"answer": answer, "confidence": confidence, "sources": sources}
