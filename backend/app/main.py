"""FastAPI application entrypoint for the RAG backend.

Responsibilities:
- Wire module boundaries together (ingestion -> embedding -> retrieval -> reranking -> generation -> verification).
- Expose API endpoints in a thin orchestration layer.
- Keep business logic out of the web layer.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator

from backend.app.config.settings import settings
from backend.app.embedding.embed import embed_chunks
from backend.app.generation.generate import generate_answer
from backend.app.ingestion.ingest import ingest_text
from backend.app.reranking.rerank import rerank_candidates
from backend.app.retrieval.retrieve import retrieve_candidates
from backend.app.storage.faiss_store import FaissStore
from backend.app.verification.verify import verify_answer

from backend.app.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="RAG System Backend")
vector_store = FaissStore()
MAX_INGEST_TEXT_LENGTH = 100_000
MAX_TOP_K = 20


@app.on_event("startup")
def load_vector_store() -> None:
    """Load persisted FAISS index at startup, or initialize empty store."""
    logger.info("Startup: MOCK_MODE=%s", settings.mock_mode)
    if not settings.mock_mode and not settings.openai_api_key.strip():
        error_message = (
            "Invalid configuration: OPENAI_API_KEY is required when MOCK_MODE is False."
        )
        logger.error(error_message)
        raise RuntimeError(error_message)

    logger.info("Startup: loading FAISS index from '%s'.", settings.faiss_index_path)
    vector_store.load_index(settings.faiss_index_path)
    stored_vectors = vector_store.vector_count()
    if stored_vectors > 0:
        logger.info("Startup: FAISS index loaded with %d vector(s).", stored_vectors)
    else:
        logger.info("Startup: FAISS index is empty.")


class QueryRequest(BaseModel):
    """Incoming request payload for the RAG query pipeline."""

    query: StrictStr
    top_k: StrictInt = 5

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: StrictStr) -> StrictStr:
        """Reject empty or whitespace-only query values."""
        if not value.strip():
            raise ValueError("query cannot be empty or whitespace.")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: StrictInt) -> StrictInt:
        """Ensure top_k is positive and within a safe upper bound."""
        if value <= 0:
            raise ValueError("top_k must be a positive integer.")
        if value > MAX_TOP_K:
            raise ValueError(f"top_k exceeds allowed limit ({MAX_TOP_K}).")
        return value


class QueryResponse(BaseModel):
    """Structured response payload from the RAG query pipeline."""

    query: str
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    answer: str
    confidence: float
    sources: List[str]


class IngestRequest(BaseModel):
    """Incoming request payload for ingestion."""

    text: StrictStr
    source: Optional[StrictStr] = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: StrictStr) -> StrictStr:
        """Reject empty or unreasonably large text payloads."""
        if not value.strip():
            raise ValueError("text cannot be empty or whitespace.")
        if len(value) > MAX_INGEST_TEXT_LENGTH:
            raise ValueError(
                f"text exceeds allowed length ({MAX_INGEST_TEXT_LENGTH} characters)."
            )
        return value


class IngestResponse(BaseModel):
    """Structured response payload for ingestion."""

    status: str
    chunks_ingested: int


class DebugChunkItem(BaseModel):
    """Inspectable chunk metadata entry."""

    chunk_id: Optional[int] = None
    source: str = "unknown"
    metadata: Dict[str, Any]


class DebugChunksResponse(BaseModel):
    """Response payload for debug chunk inspection."""

    chunks: List[DebugChunkItem]
    total_chunks: int


class HealthResponse(BaseModel):
    """Response payload for service health checks."""

    status: str
    mock_mode: bool
    vector_count: int
    index_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return current service and vector-store health status."""
    vector_count = vector_store.vector_count()
    index_loaded = vector_store.is_index_loaded()
    logger.info(
        "Health check requested: mock_mode=%s, vector_count=%d, index_loaded=%s",
        settings.mock_mode,
        vector_count,
        index_loaded,
    )
    return HealthResponse(
        status="ok",
        mock_mode=settings.mock_mode,
        vector_count=vector_count,
        index_loaded=index_loaded,
    )


@app.post("/ingest", response_model=IngestResponse)
def run_ingestion_pipeline(payload: IngestRequest) -> IngestResponse:
    """Ingest text, embed chunks, and store vectors in FAISS."""
    logger.info("Ingest Step 1/3: Chunking input text.")
    if payload.source:
        chunks = ingest_text(text=payload.text, source=payload.source)
    else:
        chunks = ingest_text(text=payload.text)

    logger.info("Ingest Step 2/3: Embedding %d chunk(s).", len(chunks))
    vectors = embed_chunks(chunks)

    logger.info("Ingest Step 3/3: Storing vectors and metadata in FAISS.")
    vector_store.add_vectors(vectors=vectors, metadata=chunks)
    vector_store.save_index(settings.faiss_index_path)

    return IngestResponse(status="success", chunks_ingested=len(chunks))


@app.get("/debug/chunks", response_model=DebugChunksResponse)
def debug_chunks() -> DebugChunksResponse:
    """Read-only endpoint to inspect stored chunk metadata."""
    logger.info("Debug: reading stored chunk metadata.")
    metadata_entries = vector_store.get_all_metadata()
    chunk_items = [
        DebugChunkItem(
            chunk_id=entry.get("chunk_id"),
            source=str(entry.get("source", "unknown")),
            metadata=entry,
        )
        for entry in metadata_entries
    ]
    return DebugChunksResponse(chunks=chunk_items, total_chunks=len(chunk_items))


@app.post("/query", response_model=QueryResponse)
def run_query_pipeline(payload: QueryRequest) -> QueryResponse:
    """Run retrieval, reranking, generation, and verification for a query."""
    logger.info("Step 1/5: Received query.")

    logger.info("Step 2/5: Retrieving chunks from FAISS.")
    retrieved_chunks = retrieve_candidates(
        query=payload.query,
        vector_store=vector_store,
        top_k=payload.top_k,
    )

    logger.info("Step 3/5: Reranking retrieved chunks.")
    reranked_chunks = rerank_candidates(query=payload.query, candidates=retrieved_chunks)

    logger.info("Step 4/5: Generating answer from selected context.")
    answer = generate_answer(query=payload.query, context_chunks=reranked_chunks)

    logger.info("Step 5/5: Verifying generated answer against context.")
    verification = verify_answer(answer=answer, context_chunks=reranked_chunks)

    return QueryResponse(
        query=payload.query,
        retrieved_chunks=retrieved_chunks,
        reranked_chunks=reranked_chunks,
        answer=verification["answer"],
        confidence=verification["confidence"],
        sources=verification["sources"],
    )
