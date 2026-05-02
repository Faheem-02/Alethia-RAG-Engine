"""FAISS storage adapter.

Responsibilities:
- Encapsulate FAISS index creation and in-memory vector operations.
- Provide vector add/search operations for retrieval.
- Keep storage-specific behavior isolated behind a simple interface.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

class FaissStore:
    """In-memory FAISS-backed vector store adapter."""

    def __init__(self, dimension: Optional[int] = None) -> None:
        self._dimension = dimension
        self._index: Optional[faiss.IndexFlatL2] = None
        self._metadata: List[Dict[str, Any]] = []

    @staticmethod
    def _metadata_file_path(index_file_path: str) -> Path:
        """Return metadata sidecar file path for a FAISS index file."""
        index_path = Path(index_file_path)
        return Path(f"{index_path}.meta.json")

    def _ensure_index(self, dimension: int) -> None:
        """Create in-memory FAISS index if needed."""
        if self._index is None:
            self._dimension = dimension
            self._index = faiss.IndexFlatL2(dimension)
            logger.info("Initialized in-memory FAISS IndexFlatL2 (dimension=%d).", dimension)
            return
        if self._dimension != dimension:
            raise ValueError(
                f"Vector dimension mismatch. Expected {self._dimension}, got {dimension}."
            )

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """Add vectors and associated metadata to the in-memory FAISS index."""
        if not vectors:
            logger.info("No vectors provided to add_vectors; skipping.")
            return
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have the same length.")

        vector_array = np.asarray(vectors, dtype=np.float32)
        if vector_array.ndim != 2:
            raise ValueError("vectors must be a 2D array-like structure.")

        self._ensure_index(dimension=vector_array.shape[1])
        self._index.add(vector_array)
        self._metadata.extend(metadata)

        logger.info("Added %d vector(s) to FAISS index.", vector_array.shape[0])

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Perform similarity search and return matches with metadata."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if self._index is None or self._index.ntotal == 0:
            logger.info("FAISS index is empty; returning no search results.")
            return []

        query_array = np.asarray([query_vector], dtype=np.float32)
        if query_array.shape[1] != self._dimension:
            raise ValueError(
                f"Query dimension mismatch. Expected {self._dimension}, got {query_array.shape[1]}."
            )

        search_k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_array, search_k)

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            distance = float(distances[0][rank])
            results.append(
                {
                    "vector_id": int(idx),
                    "metadata": self._metadata[int(idx)],
                    "distance": distance,
                    "similarity_score": 1.0 / (1.0 + distance),
                }
            )

        logger.info("Search returned %d result(s) for top_k=%d.", len(results), top_k)
        return results

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Return a read-only copy of all stored metadata entries."""
        return [dict(item) for item in self._metadata]

    def vector_count(self) -> int:
        """Return the number of vectors currently available in the index."""
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def is_index_loaded(self) -> bool:
        """Return whether a FAISS index object is currently loaded in memory."""
        return self._index is not None

    def save_index(self, file_path: str) -> None:
        """Persist FAISS index and metadata to local files."""
        if self._index is None:
            logger.info("No FAISS index to save; skipping save operation.")
            return

        index_path = Path(file_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = self._metadata_file_path(file_path)

        faiss.write_index(self._index, str(index_path))
        metadata_payload = {
            "dimension": self._dimension,
            "metadata": self._metadata,
        }
        metadata_path.write_text(
            json.dumps(metadata_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Saved FAISS index to '%s' and metadata to '%s'.",
            str(index_path),
            str(metadata_path),
        )

    def load_index(self, file_path: str) -> None:
        """Load FAISS index and metadata from local files if available."""
        index_path = Path(file_path)
        metadata_path = self._metadata_file_path(file_path)

        if not index_path.exists():
            self._index = None
            self._dimension = None
            self._metadata = []
            logger.info(
                "No existing FAISS index found at '%s'. Initialized new in-memory store.",
                str(index_path),
            )
            return

        self._index = faiss.read_index(str(index_path))
        self._dimension = int(self._index.d)

        loaded_metadata: List[Dict[str, Any]] = []
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            loaded_metadata = payload.get("metadata", [])
            if not isinstance(loaded_metadata, list):
                loaded_metadata = []
                logger.warning(
                    "Metadata file '%s' is invalid. Using empty metadata list.",
                    str(metadata_path),
                )
        else:
            logger.warning(
                "Metadata file '%s' not found. Using empty metadata placeholders.",
                str(metadata_path),
            )

        vector_count = int(self._index.ntotal)
        if len(loaded_metadata) < vector_count:
            loaded_metadata.extend({} for _ in range(vector_count - len(loaded_metadata)))
        elif len(loaded_metadata) > vector_count:
            loaded_metadata = loaded_metadata[:vector_count]

        self._metadata = loaded_metadata
        logger.info(
            "Loaded FAISS index from '%s' with %d vector(s).",
            str(index_path),
            vector_count,
        )
