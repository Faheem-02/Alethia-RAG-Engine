"""Reranking module.

Responsibilities:
- Reorder retrieved candidates by relevance quality.
- Keep reranking strategy isolated from retrieval and generation.
- Allow no-op behavior when reranking is disabled.
"""

import re
from typing import Any, Dict, List

from backend.app.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens."""
    return re.findall(r"\w+", text.lower())


def _heuristic_relevance_score(query: str, candidate: Dict[str, Any]) -> float:
    """Combine similarity score with simple query-term overlap."""
    base_similarity = float(candidate.get("similarity_score", 0.0))
    candidate_text = str(candidate.get("text", ""))

    query_terms = set(_tokenize(query))
    if not query_terms:
        return base_similarity

    candidate_terms = set(_tokenize(candidate_text))
    overlap_count = len(query_terms.intersection(candidate_terms))
    overlap_ratio = overlap_count / len(query_terms)

    # Weighted simple heuristic: prioritize vector similarity with light lexical boost.
    return (0.8 * base_similarity) + (0.2 * overlap_ratio)


def rerank_candidates(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank retrieval candidates for final context selection."""
    logger.info("Reranking %d candidate(s) for query: %s", len(candidates), query)

    scored_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        rerank_score = _heuristic_relevance_score(query=query, candidate=candidate)
        candidate_with_score = dict(candidate)
        candidate_with_score["rerank_score"] = rerank_score
        scored_candidates.append(candidate_with_score)

    scored_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    return scored_candidates
