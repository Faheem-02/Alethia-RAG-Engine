"""Embedding module.

Responsibilities:
- Convert text chunks into vector embeddings.
- Encapsulate provider-specific embedding calls (OpenAI by default).
- Support mock outputs for offline or keyless testing.
"""

import random
from typing import Any, Dict, Iterable, List

from backend.app.config.settings import settings
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)
MOCK_VECTOR_SIZE = 16
DEFAULT_BATCH_SIZE = 64
FALLBACK_ERROR_MARKERS = (
    "insufficient_quota",
    "quota",
    "out of credits",
    "rate limit",
    "429",
    "billing",
    "limit reached",
)


def _iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield fixed-size batches from a list."""
    for start_idx in range(0, len(items), batch_size):
        yield items[start_idx : start_idx + batch_size]


def _mock_embed_texts(texts: List[str]) -> List[List[float]]:
    """Return fixed-size random vectors for each text in mock mode."""
    rng = random.Random(settings.mock_seed) if settings.mock_seed is not None else random
    vectors: List[List[float]] = []
    for _ in texts:
        vectors.append([rng.random() for _ in range(MOCK_VECTOR_SIZE)])
    return vectors


def _openai_embed_texts(texts: List[str], batch_size: int) -> List[List[float]]:
    """Embed texts using the OpenAI embeddings API."""
    from openai import OpenAI

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for API-first embeddings.")

    client = OpenAI(api_key=settings.openai_api_key)
    all_vectors: List[List[float]] = []

    for batch in _iter_batches(texts, batch_size=batch_size):
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        all_vectors.extend(item.embedding for item in response.data)

    return all_vectors


def _is_fallback_eligible_error(error: Exception) -> bool:
    """Return True when provider error indicates temporary/billing limits."""
    message = str(error).lower()
    return any(marker in message for marker in FALLBACK_ERROR_MARKERS)


def embed_texts(list_of_texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[float]]:
    """Embed a list of texts with batch processing support."""
    if not list_of_texts:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    texts = [text if text is not None else "" for text in list_of_texts]

    if not settings.openai_api_key.strip():
        if settings.mock_mode:
            logger.warning(
                "OPENAI_API_KEY missing; using mock embeddings fallback (seed=%s).",
                settings.mock_seed,
            )
            return _mock_embed_texts(texts)
        raise ValueError("OPENAI_API_KEY is required for embeddings.")

    logger.info(
        "Embedding %d text(s) with OpenAI model '%s' (batch_size=%d).",
        len(texts),
        settings.embedding_model,
        batch_size,
    )
    try:
        return _openai_embed_texts(texts=texts, batch_size=batch_size)
    except Exception as error:
        if settings.mock_mode and _is_fallback_eligible_error(error):
            logger.warning(
                "Embedding API limit/billing error detected; falling back to mock embeddings: %s",
                error,
            )
            return _mock_embed_texts(texts)
        raise


def embed_chunks(chunks: List[Dict[str, Any]], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[float]]:
    """Compatibility wrapper to embed chunk payloads based on their text field."""
    texts = [str(chunk.get("text", "")) for chunk in chunks]
    return embed_texts(list_of_texts=texts, batch_size=batch_size)
