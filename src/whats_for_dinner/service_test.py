import json

import pytest
from haystack import Document

from whats_for_dinner.errors import GenerationError, RetrievalError
from whats_for_dinner.generation_test import VALID_DECISION, FakeChatGenerator
from whats_for_dinner.service import RecommendationService


class FakeTextEmbedder:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def run(self, text: str) -> dict[str, list[float]]:
        if self._error:
            raise self._error
        return {"embedding": [0.1, 0.2, 0.3]}


class FakeRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def run(self, query_embedding: list[float]) -> dict[str, list[Document]]:
        return {"documents": self._documents}


def _build_service(
    documents: list[Document] | None = None,
    embedder_error: Exception | None = None,
    generator_reply: str | None = None,
    generator_error: Exception | None = None,
) -> RecommendationService:
    default_documents = [
        Document(
            id="01",
            content="chicken stir fry recipe",
            meta={"title": "Quick Chicken Stir-Fry"},
            score=0.9,
        )
    ]
    return RecommendationService(
        text_embedder=FakeTextEmbedder(error=embedder_error),
        retriever=FakeRetriever(documents if documents is not None else default_documents),
        chat_generator=FakeChatGenerator(
            reply_text=generator_reply
            if generator_reply is not None
            else json.dumps(VALID_DECISION),
            error=generator_error,
        ),
        chat_model="gpt-4o",
        embedding_model="text-embedding-3-small",
    )


def test_recommend_returns_markdown_and_decision_summary() -> None:
    service = _build_service()

    response = service.recommend("I have chicken and soy sauce")

    assert response.recipe == VALID_DECISION["markdown"]
    assert response.decision.selected_recipe == VALID_DECISION["selected_recipe_title"]
    assert response.decision.matched_ingredients == VALID_DECISION["matched_ingredients"]
    assert response.decision.missing_ingredients == VALID_DECISION["missing_ingredients"]
    assert response.decision.is_reasonable_match is True


def test_recommend_raises_retrieval_error_on_embedding_failure() -> None:
    service = _build_service(embedder_error=TimeoutError("openai timed out"))

    with pytest.raises(RetrievalError):
        service.recommend("I have chicken")


def test_recommend_raises_generation_error_on_malformed_reply() -> None:
    service = _build_service(generator_reply="not json")

    with pytest.raises(GenerationError):
        service.recommend("I have chicken")


def test_recommend_raises_generation_error_on_api_failure() -> None:
    service = _build_service(generator_error=ConnectionError("network down"))

    with pytest.raises(GenerationError):
        service.recommend("I have chicken")


def test_recommend_canonicalizes_title_from_matched_candidate() -> None:
    # The candidate's real title differs from whatever text the LLM echoed back as
    # selected_recipe_title - the response must reflect the candidate, not the LLM's copy.
    candidate = Document(
        id="01", content="chicken stir fry recipe", meta={"title": "Real Canonical Title"}
    )
    service = _build_service(documents=[candidate])

    response = service.recommend("I have chicken and soy sauce")

    assert response.decision.selected_recipe == "Real Canonical Title"


def test_recommend_raises_generation_error_on_hallucinated_recipe_id() -> None:
    hallucinated = {**VALID_DECISION, "selected_recipe_id": "99"}
    service = _build_service(generator_reply=json.dumps(hallucinated))

    with pytest.raises(GenerationError):
        service.recommend("I have chicken and soy sauce")
