"""FastAPI application: component wiring, startup ingestion, and the thin API route.

All Haystack/OpenAI components are blocking (this pinned Haystack version has
no async document store or embedder - see README "Design decisions"), so the
one request-path call is offloaded with asyncio.to_thread instead of forcing
a partially-async pipeline for no real benefit.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from whats_for_dinner.config import get_settings
from whats_for_dinner.errors import GenerationError, RetrievalError
from whats_for_dinner.ingestion import ingest_recipes
from whats_for_dinner.models import RecommendRequest, RecommendResponse
from whats_for_dinner.service import RecommendationService

# A fresh, never-formatted record's own attributes, plus "message"/"asctime"
# which Formatter.format() adds as a side effect - used below to tell "a field
# from extra={...}" apart from every LogRecord's normal built-in attributes.
_BASE_LOG_RECORD_KEYS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
}


class _StructuredFormatter(logging.Formatter):
    """Appends logger.info(..., extra={...}) fields as JSON.

    service.py's request trace (candidate ids/scores, matched/missing
    ingredients, decision_reason, ...) is passed via `extra`, which attaches
    it to the LogRecord but - with the standard library's default
    Formatter - never gets printed. Without this, the "useful logging"
    CONVENTIONS.md asks for is only useful to something that reads LogRecord
    objects directly, not to a person watching the console.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = {k: v for k, v in vars(record).items() if k not in _BASE_LOG_RECORD_KEYS}
        return f"{base} {json.dumps(extra, default=str)}" if extra else base


_log_handler = logging.StreamHandler()
_log_handler.setFormatter(_StructuredFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    openai_api_key = Secret.from_token(settings.openai_api_key)

    document_store = PgvectorDocumentStore(
        connection_string=Secret.from_token(settings.database_url),
        table_name="recipes",
        embedding_dimension=settings.embedding_dimension,
    )
    document_embedder = OpenAIDocumentEmbedder(
        api_key=openai_api_key, model=settings.openai_embedding_model
    )

    ingested_count = await asyncio.to_thread(
        ingest_recipes, document_store, document_embedder, settings.recipe_data_path
    )
    logger.info("Startup ingestion complete: %d new recipe(s) embedded", ingested_count)

    text_embedder = OpenAITextEmbedder(
        api_key=openai_api_key, model=settings.openai_embedding_model
    )
    retriever = PgvectorEmbeddingRetriever(
        document_store=document_store, top_k=settings.retrieval_top_k
    )
    chat_generator = OpenAIChatGenerator(api_key=openai_api_key, model=settings.openai_chat_model)

    app.state.recommendation_service = RecommendationService(
        text_embedder=text_embedder,
        retriever=retriever,
        chat_generator=chat_generator,
        chat_model=settings.openai_chat_model,
        embedding_model=settings.openai_embedding_model,
    )
    yield


app = FastAPI(title="What's for Dinner", lifespan=lifespan)


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(_request: Request, exc: RetrievalError) -> JSONResponse:
    logger.error("Retrieval error: %s", exc)
    return JSONResponse(
        status_code=502, content={"detail": "Recipe retrieval is temporarily unavailable."}
    )


@app.exception_handler(GenerationError)
async def generation_error_handler(_request: Request, exc: GenerationError) -> JSONResponse:
    logger.error("Generation error: %s", exc)
    return JSONResponse(
        status_code=502, content={"detail": "Recipe recommendation is temporarily unavailable."}
    )


@app.post("/recommend_recipe")
async def recommend_recipe(request: RecommendRequest) -> RecommendResponse:
    """Recommend one recipe from the corpus for the given free-text ingredient request."""
    service: RecommendationService = app.state.recommendation_service
    return await asyncio.to_thread(service.recommend, request.text)
