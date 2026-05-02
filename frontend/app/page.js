"use client";

import { useState } from "react";

export default function Home() {
  const [ingestText, setIngestText] = useState("");
  const [queryText, setQueryText] = useState("");
  const [ingestLoading, setIngestLoading] = useState(false);
  const [queryLoading, setQueryLoading] = useState(false);
  const [ingestStatus, setIngestStatus] = useState("");
  const [error, setError] = useState("");

  const [answer, setAnswer] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [sources, setSources] = useState([]);
  const [retrievedChunks, setRetrievedChunks] = useState([]);
  const [rerankedChunks, setRerankedChunks] = useState([]);

  const clearQueryResults = () => {
    setAnswer("");
    setConfidence(null);
    setSources([]);
    setRetrievedChunks([]);
    setRerankedChunks([]);
  };

  const handleIngest = async () => {
    setError("");
    setIngestStatus("");

    if (!ingestText.trim()) {
      setError("Please enter text before ingesting.");
      return;
    }

    setIngestLoading(true);
    try {
      const response = await fetch("/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: ingestText }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.detail || "Ingest failed.");
      }

      setIngestStatus(`Ingested ${data.chunks_ingested ?? 0} chunk(s).`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected ingest error.");
    } finally {
      setIngestLoading(false);
    }
  };

  const handleAsk = async () => {
    setError("");
    clearQueryResults();

    if (!queryText.trim()) {
      setError("Please enter a query.");
      return;
    }

    setQueryLoading(true);
    try {
      const response = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText, top_k: 5 }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.detail || "Query failed.");
      }

      setAnswer(data.answer || "");
      setConfidence(typeof data.confidence === "number" ? data.confidence : null);
      setSources(Array.isArray(data.sources) ? data.sources : []);
      setRetrievedChunks(
        Array.isArray(data.retrieved_chunks) ? data.retrieved_chunks : []
      );
      setRerankedChunks(
        Array.isArray(data.reranked_chunks) ? data.reranked_chunks : []
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected query error.");
    } finally {
      setQueryLoading(false);
    }
  };

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-6 p-6">
      <h1 className="text-2xl font-semibold">RAG Frontend Demo</h1>

      <section className="rounded border p-4">
        <h2 className="mb-3 text-lg font-medium">Upload Section</h2>
        <textarea
          value={ingestText}
          onChange={(event) => setIngestText(event.target.value)}
          placeholder="Paste text to ingest"
          className="min-h-32 w-full rounded border p-2"
        />
        <div className="mt-3 flex items-center gap-3">
          <button
            onClick={handleIngest}
            disabled={ingestLoading}
            className="rounded border px-3 py-2 disabled:opacity-60"
          >
            {ingestLoading ? "Ingesting..." : "Ingest"}
          </button>
          {ingestStatus ? <p className="text-sm">{ingestStatus}</p> : null}
        </div>
      </section>

      <section className="rounded border p-4">
        <h2 className="mb-3 text-lg font-medium">Query Section</h2>
        <div className="flex gap-3">
          <input
            type="text"
            value={queryText}
            onChange={(event) => setQueryText(event.target.value)}
            placeholder="Ask a question"
            className="w-full rounded border p-2"
          />
          <button
            onClick={handleAsk}
            disabled={queryLoading}
            className="rounded border px-4 py-2 disabled:opacity-60"
          >
            {queryLoading ? "Asking..." : "Ask"}
          </button>
        </div>
      </section>

      {error ? (
        <section className="rounded border border-red-400 bg-red-50 p-4 text-red-700">
          {error}
        </section>
      ) : null}

      <section className="rounded border p-4">
        <h2 className="mb-3 text-lg font-medium">Answer Section</h2>
        <p className="mb-2">
          <span className="font-medium">Answer:</span> {answer || "-"}
        </p>
        <p className="mb-2">
          <span className="font-medium">Confidence:</span>{" "}
          {confidence === null ? "-" : confidence.toFixed(3)}
        </p>
        <p>
          <span className="font-medium">Sources:</span>{" "}
          {sources.length ? sources.join(", ") : "-"}
        </p>
      </section>

      <section className="rounded border p-4">
        <h2 className="mb-3 text-lg font-medium">Debug Section</h2>
        <div className="mb-4">
          <h3 className="mb-2 font-medium">Retrieved Chunks</h3>
          <pre className="overflow-auto rounded bg-gray-100 p-3 text-xs">
            {JSON.stringify(retrievedChunks, null, 2)}
          </pre>
        </div>
        <div>
          <h3 className="mb-2 font-medium">Reranked Chunks</h3>
          <pre className="overflow-auto rounded bg-gray-100 p-3 text-xs">
            {JSON.stringify(rerankedChunks, null, 2)}
          </pre>
        </div>
      </section>
    </main>
  );
}
