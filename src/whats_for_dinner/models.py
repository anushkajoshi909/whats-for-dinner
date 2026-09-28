"""Pydantic models shared across the recommendation pipeline.

One module for all of these on purpose: the request/response contract,
the retrieval trace, and the structured LLM decision are small and tightly
related, and PIPELINE.md asks that the core recommendation data not be
scattered across unrelated files.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RecommendRequest(BaseModel):
    """Public request body for POST /recommend_recipe."""

    text: str = Field(
        min_length=1, description="Free-form description of available ingredients/constraints."
    )

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return value


class RetrievedCandidate(BaseModel):
    """One pgvector candidate, with enough retrieval metadata to trace/evaluate it."""

    recipe_id: str
    title: str
    content: str
    rank: int
    score: float | None


class RecommendationDecision(BaseModel):
    """Structured GPT-4o output. This is the inspectable core of the system.

    extra="forbid" is required, not stylistic: it is what makes Pydantic's
    generated JSON schema include `additionalProperties: false`, which the
    OpenAI strict json_schema response format requires.
    """

    model_config = ConfigDict(extra="forbid")

    selected_recipe_id: str
    selected_recipe_title: str
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    assumed_pantry_staples: list[str]
    constraint_conflicts: list[str]
    decision_reason: str
    is_reasonable_match: bool
    markdown: str


class DecisionSummary(BaseModel):
    """Trimmed-down decision exposed on the public API response."""

    selected_recipe: str
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    assumed_pantry_staples: list[str]
    constraint_conflicts: list[str]
    is_reasonable_match: bool


class RecommendResponse(BaseModel):
    """Public response body for POST /recommend_recipe."""

    recipe: str
    decision: DecisionSummary


class RecommendationTrace(BaseModel):
    """Everything needed to reconstruct/debug/evaluate one recommendation.

    Not returned by the API (would make the public contract cumbersome per
    PIPELINE.md); logged as a single structured record per request instead.
    """

    request_id: str
    original_text: str
    candidates: list[RetrievedCandidate]
    decision: RecommendationDecision
    chat_model: str
    embedding_model: str
    latency_ms: float
