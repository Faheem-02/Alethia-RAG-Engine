"""Generation module.

Responsibilities:
- Build a grounded prompt from reranked context.
- Call the text generation provider (OpenAI by default).
- Support mock responses for testing and local development.
"""

from typing import Any, Dict, List

from backend.app.config.settings import settings
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)
INSUFFICIENT_CONTEXT_RESPONSE = (
    "I do not have enough information in the provided context to answer this query."
)
FALLBACK_ERROR_MARKERS = (
    "insufficient_quota",
    "quota",
    "out of credits",
    "rate limit",
    "429",
    "billing",
    "limit reached",
)


def _format_context(context_chunks: List[Dict[str, Any]]) -> str:
    """Convert context chunk payloads into a prompt-ready text block."""
    lines: List[str] = []
    for idx, chunk in enumerate(context_chunks):
        chunk_id = chunk.get("chunk_id", idx)
        source = chunk.get("source", "unknown")
        text = str(chunk.get("text", "")).strip()
        lines.append(f"[chunk_id={chunk_id} | source={source}] {text}")
    return "\n".join(lines)


def _build_messages(query: str, context_text: str) -> List[Dict[str, str]]:
    """Build strict grounded prompt messages."""
    return [
        {
            "role": "system",
            "content": (
                "You are a retrieval-augmented assistant. "
                "Answer using ONLY the provided context. "
                "Do not use outside knowledge. "
                "If the context is insufficient, reply exactly: "
                f"'{INSUFFICIENT_CONTEXT_RESPONSE}'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query:\n{query}\n\n"
                f"Context:\n{context_text}\n\n"
                "Return only the answer text."
            ),
        },
    ]


def _mock_generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Build a deterministic, context-aware fallback answer."""
    if not context_chunks:
        return "The system does not have enough information to answer this query."

    normalized_query = query.strip().rstrip("?.!").lower()
    selected_chunks = context_chunks[:3]

    extracted_texts: List[str] = []
    for chunk in selected_chunks:
        text = str(chunk.get("text", "")).strip()
        if text:
            extracted_texts.append(text)

    if not extracted_texts:
        return "The system does not have enough information to answer this query."

    joined_context = " ".join(extracted_texts)
    normalized_context = " ".join(joined_context.split())

    sentence_parts = [
        part.strip(" \"'")
        for part in normalized_context.replace("\n", " ").split(".")
        if part.strip()
    ]
    summary_sentences = sentence_parts[:3]
    if not summary_sentences:
        summary_sentences = [normalized_context[:240].strip()]

    summary_text = ". ".join(summary_sentences).strip()
    if summary_text and not summary_text.endswith("."):
        summary_text = f"{summary_text}."
    if not summary_text:
        return "The system does not have enough information to answer this query."

    if normalized_query.startswith(("how ", "why ", "what ", "when ", "where ", "who ")):
        return summary_text

    return f"Based on the retrieved context, {summary_text[0].lower()}{summary_text[1:]}"


def _is_fallback_eligible_error(error: Exception) -> bool:
    """Return True when provider error indicates temporary/billing limits."""
    message = str(error).lower()
    return any(marker in message for marker in FALLBACK_ERROR_MARKERS)


def generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Generate an answer using grounded context."""
    if not context_chunks:
        logger.info("No context chunks available for generation.")
        return INSUFFICIENT_CONTEXT_RESPONSE

    logger.info(
        "Generating answer with model '%s' using %d context chunk(s).",
        settings.generation_model,
        len(context_chunks),
    )
    if not settings.openai_api_key.strip():
        if settings.mock_mode:
            logger.warning("OPENAI_API_KEY missing; using mock generation fallback.")
            return _mock_generate_answer(query=query, context_chunks=context_chunks)
        raise ValueError("OPENAI_API_KEY is required for answer generation.")

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    context_text = _format_context(context_chunks)
    try:
        response = client.chat.completions.create(
            model=settings.generation_model,
            messages=_build_messages(query=query, context_text=context_text),
            temperature=0.0,
        )
    except Exception as error:
        if settings.mock_mode and _is_fallback_eligible_error(error):
            logger.warning(
                "Generation API limit/billing error detected; falling back to mock answer: %s",
                error,
            )
            return _mock_generate_answer(query=query, context_chunks=context_chunks)
        raise

    answer = response.choices[0].message.content or ""
    return answer.strip() if answer.strip() else INSUFFICIENT_CONTEXT_RESPONSE
