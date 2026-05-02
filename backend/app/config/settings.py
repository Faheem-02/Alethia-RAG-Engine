"""Centralized configuration for the RAG backend.

Responsibilities:
- Store API keys and runtime settings in one place.
- Provide toggles for mock fallback when provider limits are hit.
- Keep provider-specific values replaceable.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


def _parse_bool(value: str, default: bool = False) -> bool:
    """Parse boolean-like strings from environment variables."""
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "t", "yes", "y", "on"}


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    """Parse optional integer values from environment variables."""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return int(normalized)


@dataclass
class Settings:
    """Application settings container loaded from environment variables."""

    openai_api_key: str
    embedding_model: str
    generation_model: str
    mock_mode: bool
    mock_seed: Optional[int]
    faiss_index_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            generation_model=os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
            mock_mode=_parse_bool(os.getenv("MOCK_MODE"), default=True),
            mock_seed=_parse_optional_int(os.getenv("MOCK_SEED")),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH", "data/faiss.index"),
        )


settings = Settings.from_env()
