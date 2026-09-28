"""Tests for the evaluation runner's own logic, not the live pipeline.

main() needs a real database and OpenAI key; these exercise the pure,
Haystack-component-agnostic functions with the same small fakes the
production test suite already uses, so nothing here makes a real API/DB call.
"""

import json

from haystack import Document
from run_eval import EvalRecord, _evaluate_one

from whats_for_dinner.generation_test import VALID_DECISION, FakeChatGenerator
from whats_for_dinner.service_test import FakeRetriever, FakeTextEmbedder


def _build_record(**overrides: object) -> EvalRecord:
    defaults: dict[str, object] = {
        "query": "I have chicken and soy sauce",
        "relevant_recipe_ids": ["01"],
        "preferred_recipe_id": "01",
        "expected_matched_ingredients": ["chicken", "soy sauce"],
        "expected_missing_ingredients": [],
        "expected_constraint_conflicts": [],
        "expected_is_reasonable_match": True,
    }
    return EvalRecord.model_validate({**defaults, **overrides})


def test_evaluate_one_reports_selection_failure_on_hallucinated_recipe_id() -> None:
    # Same invariant production enforces (service.py's validate_selection): a
    # selected_recipe_id that isn't one of the retrieved candidates must not be
    # silently accepted here either - it should be a labeled failure, not a
    # crash that takes down the whole evaluation run over one bad reply.
    candidates = [
        Document(
            id="01", content="chicken stir fry recipe", meta={"title": "Quick Chicken Stir-Fry"}
        )
    ]
    hallucinated_reply = json.dumps({**VALID_DECISION, "selected_recipe_id": "99"})

    result = _evaluate_one(
        _build_record(),
        text_embedder=FakeTextEmbedder(),
        retriever=FakeRetriever(candidates),
        chat_generator=FakeChatGenerator(reply_text=hallucinated_reply),
        id_map={"01": "01"},
    )

    assert result.decision is None
    assert result.failure_stage == "selection failure"


def test_evaluate_one_accepts_a_valid_selection() -> None:
    candidates = [
        Document(
            id="01", content="chicken stir fry recipe", meta={"title": "Quick Chicken Stir-Fry"}
        )
    ]

    result = _evaluate_one(
        _build_record(),
        text_embedder=FakeTextEmbedder(),
        retriever=FakeRetriever(candidates),
        chat_generator=FakeChatGenerator(reply_text=json.dumps(VALID_DECISION)),
        id_map={"01": "01"},
    )

    assert result.decision is not None
    assert result.decision.selected_recipe_id == "01"
    assert result.failure_stage == "pass"
