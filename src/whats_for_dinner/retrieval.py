"""Query embedding + pgvector candidate retrieval.

Retrieval only answers "which recipes are semantically related to this
request?" - it deliberately does not decide feasibility or apply a hard
similarity cutoff (PIPELINE.md section 12). That decision belongs to generation.py.
"""

import logging
from typing import Protocol

from haystack import Document

from whats_for_dinner.errors import RetrievalError
from whats_for_dinner.models import RetrievedCandidate

logger = logging.getLogger(__name__)


class TextEmbedder(Protocol):
    """The one OpenAITextEmbedder method this module depends on.

    A structural interface, not a wrapper: real Haystack components satisfy
    it as-is, and tests can pass a plain fake without needing a fake that
    subclasses OpenAITextEmbedder.
    """

    def run(self, text: str) -> dict[str, list[float]]: ...


class CandidateRetriever(Protocol):
    """The one PgvectorEmbeddingRetriever method this module depends on."""

    def run(self, query_embedding: list[float]) -> dict[str, list[Document]]: ...


def retrieve_candidates(
    text_embedder: TextEmbedder,
    retriever: CandidateRetriever,
    request_text: str,
) -> list[RetrievedCandidate]:
    """Embed the full user request and fetch the top-k most similar recipes.

    The complete request is embedded, not just extracted ingredient names -
    constraints like "no cheese" or "quick" carry semantic signal that a
    naive ingredient-only query would discard (PIPELINE.md section 11).

    Raises:
        RetrievalError: if query embedding or pgvector retrieval fails.
    """
    try:
        query_embedding = text_embedder.run(text=request_text)["embedding"]
        documents = retriever.run(query_embedding=query_embedding)["documents"]
    except Exception as exc:  # openai/psycopg raise varied exception types
        raise RetrievalError("Failed to retrieve candidate recipes") from exc

    return [
        RetrievedCandidate(
            recipe_id=document.id,
            title=document.meta.get("title", "Untitled Recipe"),
            content=document.content or "",
            rank=rank,
            score=document.score,
        )
        for rank, document in enumerate(documents, start=1)
    ]
