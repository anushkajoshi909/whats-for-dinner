"""Domain exceptions for the recommendation flow.

Kept flat and small on purpose: each stage that can fail in a way the API
needs to explain to a client gets exactly one exception type. Infrastructure
exceptions (openai, psycopg, ...) are caught close to their source and
re-raised as one of these so FastAPI never has to translate a raw driver
exception into an HTTP response.
"""


class RecommendationError(Exception):
    """Base class for all recommendation-flow failures."""


class RetrievalError(RecommendationError):
    """Query embedding or pgvector candidate retrieval failed."""


class GenerationError(RecommendationError):
    """GPT-4o call failed, or its reply didn't match the expected schema."""
