import json

import pytest
from haystack.dataclasses import ChatMessage

from whats_for_dinner.errors import GenerationError
from whats_for_dinner.generation import generate_recommendation

VALID_DECISION = {
    "selected_recipe_id": "01",
    "selected_recipe_title": "Quick Chicken Stir-Fry",
    "matched_ingredients": ["chicken", "soy sauce"],
    "missing_ingredients": ["mixed vegetables", "garlic"],
    "assumed_pantry_staples": ["vegetable oil"],
    "constraint_conflicts": [],
    "decision_reason": "Best ingredient coverage among the retrieved candidates.",
    "is_reasonable_match": True,
    "markdown": "## Quick Chicken Stir-Fry\n...",
}


class FakeChatGenerator:
    """Stands in for OpenAIChatGenerator so tests never call the real API."""

    def __init__(self, reply_text: str | None = None, error: Exception | None = None) -> None:
        self._reply_text = reply_text
        self._error = error
        self.last_generation_kwargs: dict[str, object] | None = None

    def run(
        self, messages: list[ChatMessage], *, generation_kwargs: dict[str, object]
    ) -> dict[str, list[ChatMessage]]:
        self.last_generation_kwargs = generation_kwargs
        if self._error:
            raise self._error
        assert self._reply_text is not None
        return {"replies": [ChatMessage.from_assistant(self._reply_text)]}


def test_generate_recommendation_parses_a_valid_reply() -> None:
    generator = FakeChatGenerator(reply_text=json.dumps(VALID_DECISION))

    decision = generate_recommendation(generator, messages=[ChatMessage.from_user("hi")])

    assert decision.selected_recipe_id == "01"
    assert decision.matched_ingredients == ["chicken", "soy sauce"]
    assert decision.is_reasonable_match is True


def test_generate_recommendation_requests_strict_json_schema() -> None:
    generator = FakeChatGenerator(reply_text=json.dumps(VALID_DECISION))

    generate_recommendation(generator, messages=[ChatMessage.from_user("hi")])

    assert generator.last_generation_kwargs is not None
    response_format = generator.last_generation_kwargs["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert isinstance(json_schema, dict)
    assert json_schema["strict"] is True


def test_generate_recommendation_raises_on_malformed_json() -> None:
    generator = FakeChatGenerator(reply_text="not valid json at all")

    with pytest.raises(GenerationError):
        generate_recommendation(generator, messages=[ChatMessage.from_user("hi")])


def test_generate_recommendation_raises_when_required_field_missing() -> None:
    incomplete = {k: v for k, v in VALID_DECISION.items() if k != "decision_reason"}

    generator = FakeChatGenerator(reply_text=json.dumps(incomplete))

    with pytest.raises(GenerationError):
        generate_recommendation(generator, messages=[ChatMessage.from_user("hi")])


def test_generate_recommendation_wraps_api_failures() -> None:
    generator = FakeChatGenerator(error=ConnectionError("network down"))

    with pytest.raises(GenerationError):
        generate_recommendation(generator, messages=[ChatMessage.from_user("hi")])


def test_generate_recommendation_deduplicates_pantry_staples_from_missing() -> None:
    decision_with_overlap = {
        **VALID_DECISION,
        "missing_ingredients": ["garlic", "cooking oil"],
        "assumed_pantry_staples": ["cooking oil"],
    }
    generator = FakeChatGenerator(reply_text=json.dumps(decision_with_overlap))

    decision = generate_recommendation(generator, messages=[ChatMessage.from_user("hi")])

    assert decision.missing_ingredients == ["garlic"]
    assert decision.assumed_pantry_staples == ["cooking oil"]
