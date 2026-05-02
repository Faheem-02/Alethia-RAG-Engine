"""Document ingestion module.

Responsibilities:
- Accept plain text input (initial version).
- Split content into paragraph-based chunks.
- Attach source metadata for downstream pipeline stages.
"""

from pathlib import Path
from typing import Any, Dict, List

from backend.app.utils.logger import get_logger

logger = get_logger(__name__)


def ingest_text(text: str, source: str = "text_input") -> List[Dict[str, Any]]:
    """Split text by paragraphs and return structured chunks."""
    cleaned_text = text.strip()
    if not cleaned_text:
        logger.info("Received empty text input for ingestion.")
        return []

    paragraphs = [paragraph.strip() for paragraph in cleaned_text.split("\n\n")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]

    chunks: List[Dict[str, Any]] = []
    for idx, paragraph in enumerate(paragraphs):
        chunks.append(
            {
                "text": paragraph,
                "source": source,
                "chunk_id": idx,
                "position": idx,
            }
        )

    logger.info("Ingested %d paragraph chunk(s) from source '%s'.", len(chunks), source)
    return chunks


def ingest_documents(source_path: Path) -> List[Dict[str, Any]]:
    """Placeholder for file-based ingestion (not implemented yet)."""
    raise NotImplementedError(
        "File-based ingestion is not implemented yet. Use ingest_text for initial text input."
    )
