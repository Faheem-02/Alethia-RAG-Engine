# 🧠 Aletheia RAG Engine — Explainable AI Retrieval System

A full-stack Retrieval-Augmented Generation (RAG) system designed to deliver **context-grounded, explainable answers** instead of relying on uncontrolled model memory.

---

## 🚀 Overview

This project implements a **complete, production-style RAG pipeline** with a focus on:

* Reliability over hallucination
* Explainability over black-box outputs
* System design over simple workflows

Unlike typical chatbot demos, this system explicitly separates:

> **Knowledge → Retrieval → Reasoning → Verification**

---

## 🎯 Problem It Solves

Most AI systems:

* Generate answers without grounding ❌
* Cannot explain where information comes from ❌
* Fail under API limits or missing context ❌

---

### This system solves that by:

* Retrieving relevant knowledge from stored data
* Generating answers grounded in context
* Providing confidence scores and source attribution
* Falling back safely when APIs fail

---

## 🏗️ System Architecture

### 🔹 Backend (FastAPI)

Modular pipeline:

```
Ingestion → Embedding → Storage (FAISS)
        → Retrieval → Reranking
        → Generation → Verification
```

Key characteristics:

* API-first design
* Deterministic mock fallback
* Modular separation of concerns
* Observability via debug endpoints

---

### 🔹 Frontend (Next.js)

Minimal UI for system interaction:

* Upload knowledge (ingestion)
* Ask questions (query)
* View:

  * Answer
  * Confidence score
  * Sources
  * Retrieved & reranked chunks

---

## ⚙️ Core Features

* ✅ Context-grounded answers (no blind hallucination)
* ✅ Multi-document retrieval
* ✅ Confidence scoring mechanism
* ✅ Source traceability
* ✅ API-first execution with fallback
* ✅ Deterministic mock mode (no API required)
* ✅ Debug visibility (retrieval + reranking pipeline)

---

## 🔁 Execution Flow

1. User ingests raw text
2. System chunks and embeds content
3. Vectors stored in FAISS
4. Query triggers retrieval
5. Results reranked
6. Answer generated from context
7. Output verified with confidence + sources

---

## 🧪 Example

**Input Knowledge**

```
AI systems store knowledge and answer queries using context.
```

**Query**

```
How does the system answer questions?
```

**Output**

* Context-derived answer
* Confidence score
* Source references

---

## 🛠️ Tech Stack

### Backend

* FastAPI
* FAISS (vector similarity search)
* Python

### Frontend

* Next.js (App Router)
* Tailwind CSS

---

## ▶️ Running Locally

### Backend

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload
```

---

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

### Access

```
http://localhost:3000
```

---

## 🔁 Modes

### API Mode (Primary)

* Uses OpenAI for embeddings and answer generation
* Default execution path when API key is available
* Produces higher-quality, model-driven responses

---

### Mock Fallback Mode

* Automatically activated when:

  * API key is missing and `MOCK_MODE=true`, or
  * provider errors occur (quota, rate limit, billing issues)
* Generates deterministic, context-aware responses from retrieved chunks
* Ensures system reliability even without external API access

---

### ⚙️ Behavior Summary

| Condition                    | Mode          |
| ---------------------------- | ------------- |
| Valid API key + working API  | API Mode      |
| API limit / quota exceeded   | Mock Fallback |
| No API key + MOCK_MODE=true  | Mock Mode     |
| No API key + MOCK_MODE=false | Error         |


---

## 📌 Key Design Decisions

* **API-first architecture** → aligns with real-world systems
* **Fallback mechanism** → prevents failure under constraints
* **Explainability layer** → exposes retrieval + reasoning path
* **Modular pipeline** → independently testable components

---

## 🚀 Why This Project Stands Out

This is not a simple chatbot.

It demonstrates:

* Separation of retrieval and generation
* Controlled, non-hallucinating design
* Explainable AI outputs
* Real-world system reliability patterns

---

## 📈 Future Improvements

* Semantic reranking (cross-encoder)
* Better confidence calibration
* Streaming responses
* Cloud deployment (Vercel + backend hosting)

---

## 👨‍💻 Author
Faheem
Built with a focus on **system design, explainable AI, and production-ready architecture**.
