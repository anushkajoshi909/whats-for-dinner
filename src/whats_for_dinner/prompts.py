"""The recommendation prompt and pantry-staple policy.

Kept in one obvious, easy-to-edit place per PIPELINE.md section 10 - the rules that
govern the recommendation decision should not be scattered across unrelated
files. If you want to change how GPT-4o reasons about candidates, this is
the only file you should need to touch.
"""

from haystack.dataclasses import ChatMessage

from whats_for_dinner.models import RetrievedCandidate

# Small, explicit, and easy to change (PIPELINE.md section 15). The LLM must not
# assume any missing ingredient beyond this list is "probably fine to skip".
PANTRY_STAPLES: list[str] = ["salt", "pepper", "water", "cooking oil"]

# Real recipes name "cooking oil" differently (vegetable, olive, canola, ...); listed
# explicitly here rather than left to the model's own judgment, so "assumed pantry
# staple" stays deterministic instead of depending on which synonym a recipe happens
# to use. Kept separate from PANTRY_STAPLES itself so that list stays short.
_COOKING_OIL_VARIANTS: list[str] = ["cooking oil", "vegetable oil", "olive oil", "canola oil"]

SYSTEM_PROMPT = f"""You are a recipe recommendation engine. You choose ONE recipe from a \
supplied list of candidates to recommend to a user, based only on the ingredients and \
constraints they describe.

Rules:
1. Only consider the candidate recipes supplied below. Never invent a recipe that isn't \
one of them.
2. Select the candidate that best fits the user's available ingredients and constraints.
3. Prefer the candidate with fewer missing CORE ingredients (i.e. not pantry staples).
4. Respect explicit exclusions and dietary constraints (e.g. "no cheese", "vegetarian"). \
If every candidate conflicts with a constraint, set is_reasonable_match to false.
5. matched_ingredients: recipe ingredients the user explicitly stated they have. Never \
claim the user has an ingredient they did not mention.
6. missing_ingredients: recipe ingredients required by the selected recipe that the user \
did not mention and that are NOT one of the pantry staples below.
7. assumed_pantry_staples: recipe ingredients required by the selected recipe that the \
user did not mention but that are one of these pantry staples: {", ".join(PANTRY_STAPLES)} \
(for cooking oil, any of these count as the same staple: {", ".join(_COOKING_OIL_VARIANTS)}). \
Only use this list - do not assume any other missing ingredient is a pantry staple.
8. An ingredient must appear in exactly one of matched_ingredients, missing_ingredients, or \
assumed_pantry_staples - never in more than one.
9. Do not silently drop an important missing ingredient from missing_ingredients just to \
make the recipe look more feasible.
10. constraint_conflicts: any explicit user constraint the selected recipe violates.
11. If no candidate is reasonably suitable (e.g. all conflict with a constraint, or all are \
missing too many core ingredients), set is_reasonable_match to false and select the closest \
candidate anyway so its gaps can be explained.
12. decision_reason: one or two concise sentences a developer could use to debug this \
decision. State facts (ingredient coverage, constraints), not private reasoning steps.
13. markdown: a readable, self-contained recipe recommendation in Markdown, grounded only \
in the selected candidate's text - do not add ingredients or steps it doesn't contain. If \
is_reasonable_match is false, say so plainly and explain what would be needed.
"""

_USER_PROMPT_TEMPLATE = """User request:
{request_text}

Candidate recipes (retrieved by semantic similarity - relevance only, not a feasibility \
ranking):

{candidates_block}

Choose the best candidate and produce the structured decision."""


def build_recommendation_messages(
    request_text: str, candidates: list[RetrievedCandidate]
) -> list[ChatMessage]:
    """Assemble the system + user messages sent to GPT-4o for one recommendation."""
    candidates_block = "\n\n".join(
        f"--- Candidate {candidate.rank} "
        f"(id: {candidate.recipe_id}, title: {candidate.title}) ---\n"
        f"{candidate.content}"
        for candidate in candidates
    )
    return [
        ChatMessage.from_system(SYSTEM_PROMPT),
        ChatMessage.from_user(
            _USER_PROMPT_TEMPLATE.format(
                request_text=request_text, candidates_block=candidates_block
            )
        ),
    ]
