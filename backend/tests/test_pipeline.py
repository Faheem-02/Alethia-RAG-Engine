"""Pipeline tests for module-level and API-level mock-mode behavior."""

from fastapi.testclient import TestClient

from backend.app.config.settings import settings
from backend.app.embedding.embed import MOCK_VECTOR_SIZE, embed_chunks, embed_texts
from backend.app.generation.generate import generate_answer
from backend.app.main import app
from backend.app.reranking.rerank import rerank_candidates
from backend.app.retrieval.retrieve import retrieve_candidates
from backend.app.storage.faiss_store import FaissStore
from backend.app.verification.verify import verify_answer


def test_pipeline_modules_mock_mode() -> None:
    """Test core pipeline modules without API layer."""
    settings.mock_mode = True
    settings.openai_api_key = ""
    settings.mock_seed = 42

    chunks = [
        {"chunk_id": 1, "text": "Sample chunk text one.", "source": "test", "position": 0},
        {"chunk_id": 2, "text": "Sample chunk text two.", "source": "test", "position": 1},
        {"chunk_id": 3, "text": "Sample chunk text three.", "source": "test", "position": 2},
    ]

    vectors = embed_chunks(chunks)
    assert len(vectors) == 3
    assert len(vectors[0]) == MOCK_VECTOR_SIZE

    store = FaissStore()
    store.add_vectors(vectors=vectors, metadata=chunks)

    retrieved = retrieve_candidates(
        query="What is this document about?",
        vector_store=store,
        top_k=3,
    )
    assert len(retrieved) == 3

    reranked = rerank_candidates(
        query="What is this document about?",
        candidates=retrieved,
    )

    answer = generate_answer(
        query="What is this document about?",
        context_chunks=reranked,
    )

    verification = verify_answer(
        answer=answer,
        context_chunks=reranked,
    )

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    assert any(word in answer.lower() for word in ["system", "context", "data", "chunk", "text"])
    assert 0.0 <= verification["confidence"] <= 1.0
    assert isinstance(verification["sources"], list)


def test_api_end_to_end_mock_mode(tmp_path) -> None:
    """Test full API flow using TestClient."""
    settings.mock_mode = True
    settings.openai_api_key = ""
    settings.mock_seed = 42
    settings.faiss_index_path = str(tmp_path / "test.index")

    with TestClient(app) as client:
        ingest_response = client.post(
            "/ingest",
            json={"text": "This system answers using context.", "source": "test"},
        )
        assert ingest_response.status_code == 200
        assert ingest_response.json()["chunks_ingested"] > 0

        query_response = client.post(
            "/query",
            json={"query": "How does the system answer?", "top_k": 3},
        )
        assert query_response.status_code == 200

        data = query_response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"].strip()) > 0
        assert any(
            word in data["answer"].lower()
            for word in ["system", "context", "data", "chunk", "text"]
        )

        assert "confidence" in data
        assert isinstance(data["confidence"], float)

        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0


def test_mock_embeddings_deterministic() -> None:
    """Ensure mock embeddings are deterministic with seed."""
    settings.mock_mode = True
    settings.openai_api_key = ""
    settings.mock_seed = 42

    inputs = ["alpha paragraph", "beta paragraph"]
    first = embed_texts(inputs)
    second = embed_texts(inputs)

    assert first == second
