"""Retrieval module.

Responsibilities:
- Query vector storage with a user question embedding.
- Return top-k candidate chunks and metadata.
- Keep retrieval independent from generation and verification.
"""

from typing import Any, Dict, List

from backend.app.embedding.embed import embed_texts
from backend.app.storage.faiss_store import FaissStore
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)


def _build_candidates(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize vector-store matches into retrieval candidates."""
    candidates: List[Dict[str, Any]] = []
    for result in search_results:
        metadata = result.get("metadata", {})
        candidate: Dict[str, Any] = {
            "vector_id": result.get("vector_id"),
            "distance": result.get("distance"),
            "similarity_score": result.get("similarity_score"),
            **metadata,
        }
        candidates.append(candidate)
    return candidates


def retrieve_candidates(query: str, vector_store: FaissStore, top_k: int = 5) -> List[Dict[str, Any]]:
    """Embed query and retrieve top-k candidates from FAISS."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    query_vectors = embed_texts([query])
    query_vector = query_vectors[0]

    logger.info("Searching FAISS for top_k=%d candidates.", top_k)
    search_results = vector_store.search(query_vector=query_vector, top_k=top_k)
    candidates = _build_candidates(search_results)

    logger.info("Retrieved %d candidate(s) from FAISS.", len(candidates))
    return candidates
