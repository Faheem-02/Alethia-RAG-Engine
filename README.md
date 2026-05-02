# RAG System (FastAPI + Next.js)

A modular Retrieval-Augmented Generation (RAG) demo project with:
- A Python FastAPI backend for ingestion, retrieval, generation, and verification
- A minimal Next.js frontend to interact with the pipeline
- API-first behavior with mock fallback when provider limits/credits fail

## Project Intent

This project is designed to clearly demonstrate a full RAG flow:
1. Ingest source text
2. Chunk and embed the text
3. Store vectors in FAISS
4. Retrieve and rerank relevant chunks for a query
5. Generate a grounded answer
6. Verify confidence and source support

The codebase prioritizes readability and modularity so each stage can be understood independently.

## Repository Layout

```text
rag-system/
├─ backend/
│  ├─ app/
│  │  ├─ main.py                  # FastAPI app and API endpoints
│  │  ├─ config/settings.py       # Env-driven settings
│  │  ├─ ingestion/ingest.py      # Text chunking
│  │  ├─ embedding/embed.py       # Embedding (OpenAI first, mock fallback)
│  │  ├─ retrieval/retrieve.py    # Vector retrieval
│  │  ├─ reranking/rerank.py      # Candidate reranking
│  │  ├─ generation/generate.py   # Answer generation (OpenAI first, mock fallback)
│  │  ├─ verification/verify.py   # Confidence/source verification
│  │  ├─ storage/faiss_store.py   # FAISS persistence and search
│  │  └─ utils/logger.py          # Logging helper
│  ├─ tests/
│  │  ├─ conftest.py
│  │  └─ test_pipeline.py
│  └─ requirements.txt
├─ frontend/
│  ├─ app/page.js                 # Single-page RAG UI
│  ├─ app/ingest/route.js         # Proxy to backend /ingest
│  └─ app/query/route.js          # Proxy to backend /query
└─ README.md
```

## Backend API

Base URL:
- `http://127.0.0.1:8000`

Endpoints:
- `GET /health` - service health
- `POST /ingest` - ingest raw text and create embeddings
- `POST /query` - run retrieval + reranking + generation + verification
- `GET /debug/chunks` - inspect metadata currently in FAISS

## API-First with Mock Fallback

Current behavior:
- Backend attempts OpenAI API calls first for embeddings and generation.
- Mock behavior is fallback-only when:
  - `OPENAI_API_KEY` is unavailable and `MOCK_MODE=true`, or
  - provider errors indicate quota/credit/rate-limit/billing constraints
- Non-fallback errors are raised as valid errors.

This gives real model behavior by default while keeping demos resilient when credits are exhausted.

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 18+ and npm

## Setup

### 1) Backend

From `rag-system` root:

```bash
python -m venv .venv
```

Activate virtual environment:

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create environment variables (shell or `.env` in `backend`):

```env
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-4o-mini
MOCK_MODE=true
MOCK_SEED=42
FAISS_INDEX_PATH=data/faiss.index
```

Run backend:

```bash
python -m uvicorn backend.app.main:app --reload
```

### 2) Frontend

From `rag-system/frontend`:

```bash
npm install
npm run dev
```

Open:
- `http://localhost:3000`

The frontend page includes:
- Upload section (`/ingest`)
- Query section (`/query`)
- Answer section (answer, confidence, sources)
- Debug section (retrieved chunks, reranked chunks)

## Test Commands

Backend tests:

```bash
cd backend
python -m pytest -q
```

Frontend build check:

```bash
cd frontend
npm run build
```

## Typical End-to-End Flow

1. Start backend server
2. Start frontend server
3. Paste text in Upload section and click **Ingest**
4. Ask a query in Query section and click **Ask**
5. Inspect answer and debug chunks in UI

## Notes for Contributors

- Keep module boundaries intact (ingestion/embedding/retrieval/reranking/generation/verification)
- Avoid coupling UI logic with backend internals
- Prefer deterministic behavior in tests and mock fallback responses
