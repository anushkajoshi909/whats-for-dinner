"""GPT-4o structured recommendation generation.

Haystack 2.12's OpenAIChatGenerator.run has no response_format parameter, so
structured output is requested the way the underlying openai==1.75.0 client
supports it: a strict JSON-schema response_format passed through
generation_kwargs (verified directly against the pinned versions - see
README "Design decisions"). The reply is then validated against
RecommendationDecision, so a malformed reply fails loudly as a
GenerationError rather than silently producing bad data.
"""

import logging
from typing import Protocol

from haystack.dataclasses import ChatMessage
from pydantic import ValidationError

from whats_for_dinner.errors import GenerationError
from whats_for_dinner.models import RecommendationDecision

logger = logging.getLogger(__name__)

_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "recommendation_decision",
        "schema": RecommendationDecision.model_json_schema(),
        "strict": True,
    },
}


class ChatGenerator(Protocol):
    """The one OpenAIChatGenerator method this module depends on.

    generation_kwargs is keyword-only here (matching how it's always called)
    so this only has to describe the one parameter this module uses, rather
    than the generator's full run() signature (streaming_callback, tools, ...).
    """

    def run(
        self, messages: list[ChatMessage], *, generation_kwargs: dict[str, object]
    ) -> dict[str, list[ChatMessage]]: ...


def generate_recommendation(
    chat_generator: ChatGenerator, messages: list[ChatMessage]
) -> RecommendationDecision:
    """Call GPT-4o and parse its reply into a validated RecommendationDecision.

    Raises:
        GenerationError: the API call failed, or the reply didn't match the schema.
    """
    try:
        result = chat_generator.run(
            messages=messages, generation_kwargs={"response_format": _RESPONSE_SCHEMA}
        )
    except Exception as exc:  # openai raises several distinct exception types
        raise GenerationError("GPT-4o request failed") from exc

    replies = result.get("replies", [])
    reply_text = replies[0].text if replies else None
    if not reply_text:
        raise GenerationError("GPT-4o returned no reply")

    try:
        decision = RecommendationDecision.model_validate_json(reply_text)
    except ValidationError as exc:
        raise GenerationError("GPT-4o reply did not match the expected decision schema") from exc

    return _deduplicate_pantry_staples(decision)


def _deduplicate_pantry_staples(decision: RecommendationDecision) -> RecommendationDecision:
    """Guard against the model listing the same ingredient as both missing and a pantry staple.

    The prompt instructs the model not to do this, but LLM output is not
    guaranteed to honor every instruction; this keeps the documented
    invariant (PIPELINE.md section 15) true regardless.
    """
    pantry_set = {item.lower() for item in decision.assumed_pantry_staples}
    deduplicated_missing = [
        item for item in decision.missing_ingredients if item.lower() not in pantry_set
    ]
    if deduplicated_missing == decision.missing_ingredients:
        return decision
    return decision.model_copy(update={"missing_ingredients": deduplicated_missing})
