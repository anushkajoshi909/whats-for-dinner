"""Recommendation service: the top-down business flow for one request.

This is the only place that knows the full pipeline order (retrieve ->
build prompt -> generate -> trace). The FastAPI route stays thin by
delegating everything here.
"""

import logging
import time
import uuid

from whats_for_dinner.errors import GenerationError
from whats_for_dinner.generation import ChatGenerator, generate_recommendation
from whats_for_dinner.models import (
    DecisionSummary,
    RecommendationDecision,
    RecommendationTrace,
    RecommendResponse,
    RetrievedCandidate,
)
from whats_for_dinner.prompts import build_recommendation_messages
from whats_for_dinner.retrieval import CandidateRetriever, TextEmbedder, retrieve_candidates

logger = logging.getLogger(__name__)


def _validate_selection(
    candidates: list[RetrievedCandidate], decision: RecommendationDecision
) -> RecommendationDecision:
    """Reject a decision that points at a recipe retrieval never supplied.

    Structural validity (generation.py's JSON-schema + Pydantic check) only
    guarantees the shape of the reply, not that selected_recipe_id refers to
    a real candidate - the model can still hallucinate an id. Canonicalizing
    the title from the matched candidate, rather than trusting the LLM's own
    copy of it, also makes a valid-id/wrong-title reply behave
    deterministically instead of silently displaying whatever text GPT-4o
    happened to write.
    """
    candidates_by_id = {candidate.recipe_id: candidate for candidate in candidates}
    matched_candidate = candidates_by_id.get(decision.selected_recipe_id)
    if matched_candidate is None:
        raise GenerationError(
            f"GPT-4o selected recipe id {decision.selected_recipe_id!r}, "
            "which was not among the retrieved candidates"
        )
    return decision.model_copy(update={"selected_recipe_title": matched_candidate.title})


class RecommendationService:
    """Turns a free-text ingredient request into a traceable recipe recommendation."""

    def __init__(
        self,
        text_embedder: TextEmbedder,
        retriever: CandidateRetriever,
        chat_generator: ChatGenerator,
        chat_model: str,
        embedding_model: str,
    ) -> None:
        self._text_embedder = text_embedder
        self._retriever = retriever
        self._chat_generator = chat_generator
        self._chat_model = chat_model
        self._embedding_model = embedding_model

    def recommend(self, request_text: str) -> RecommendResponse:
        """Run retrieval + GPT-4o reasoning and return the public response.

        Args:
            request_text: the user's free-form ingredient/constraint description.

        Returns:
            The Markdown recommendation plus a public decision summary.

        Raises:
            RetrievalError: query embedding or pgvector retrieval failed.
            GenerationError: the GPT-4o call failed, its reply was malformed, or it
                selected a recipe id that wasn't one of the retrieved candidates.
        """
        request_id = str(uuid.uuid4())
        started_at = time.perf_counter()

        try:
            candidates = retrieve_candidates(self._text_embedder, self._retriever, request_text)
            messages = build_recommendation_messages(request_text, candidates)
            decision = _validate_selection(
                candidates, generate_recommendation(self._chat_generator, messages)
            )
        except Exception as exc:
            # Catches RetrievalError/GenerationError (the expected failure modes -
            # see Raises below) as well as any genuinely unexpected bug in this
            # pipeline. Either way it's logged with the same request-id correlation
            # before re-raising unchanged, so no failure is invisible in the trace -
            # an unexpected exception still reaches FastAPI's default safe handling,
            # it just isn't silently missing from the logs on its way there.
            self._log_failure(request_id, request_text, time.perf_counter() - started_at, exc)
            raise

        latency_ms = (time.perf_counter() - started_at) * 1000
        trace = RecommendationTrace(
            request_id=request_id,
            original_text=request_text,
            candidates=candidates,
            decision=decision,
            chat_model=self._chat_model,
            embedding_model=self._embedding_model,
            latency_ms=latency_ms,
        )
        self._log_trace(trace)

        return RecommendResponse(
            recipe=decision.markdown,
            decision=DecisionSummary(
                selected_recipe=decision.selected_recipe_title,
                matched_ingredients=decision.matched_ingredients,
                missing_ingredients=decision.missing_ingredients,
                assumed_pantry_staples=decision.assumed_pantry_staples,
                constraint_conflicts=decision.constraint_conflicts,
                is_reasonable_match=decision.is_reasonable_match,
            ),
        )

    def _log_trace(self, trace: RecommendationTrace) -> None:
        # One structured record per request lets an engineer answer, after the
        # fact, whether a failure was retrieval or generation (PIPELINE.md section 17).
        logger.info(
            "Recommendation decision",
            extra={
                "request_id": trace.request_id,
                "candidate_ids": [c.recipe_id for c in trace.candidates],
                "candidate_titles": [c.title for c in trace.candidates],
                "candidate_scores": [c.score for c in trace.candidates],
                "selected_recipe_id": trace.decision.selected_recipe_id,
                "selected_recipe_title": trace.decision.selected_recipe_title,
                "matched_ingredients": trace.decision.matched_ingredients,
                "missing_ingredients": trace.decision.missing_ingredients,
                "assumed_pantry_staples": trace.decision.assumed_pantry_staples,
                "constraint_conflicts": trace.decision.constraint_conflicts,
                "decision_reason": trace.decision.decision_reason,
                "is_reasonable_match": trace.decision.is_reasonable_match,
                "chat_model": trace.chat_model,
                "embedding_model": trace.embedding_model,
                "latency_ms": round(trace.latency_ms, 1),
            },
        )

    def _log_failure(
        self, request_id: str, request_text: str, elapsed_seconds: float, error: Exception
    ) -> None:
        logger.warning(
            "Recommendation failed",
            extra={
                "request_id": request_id,
                "request_length": len(request_text),
                "latency_ms": round(elapsed_seconds * 1000, 1),
                "error_type": type(error).__name__,
            },
        )
