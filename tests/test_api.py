import pytest
from httpx import ASGITransport, AsyncClient

from whats_for_dinner.errors import GenerationError, RetrievalError
from whats_for_dinner.main import app
from whats_for_dinner.models import DecisionSummary, RecommendResponse

pytestmark = pytest.mark.anyio

_FAKE_RESPONSE = RecommendResponse(
    recipe="## Quick Chicken Stir-Fry\n...",
    decision=DecisionSummary(
        selected_recipe="Quick Chicken Stir-Fry",
        matched_ingredients=["chicken", "soy sauce"],
        missing_ingredients=["mixed vegetables", "garlic"],
        assumed_pantry_staples=["vegetable oil"],
        constraint_conflicts=[],
        is_reasonable_match=True,
    ),
)


class StubService:
    """Injected in place of the real RecommendationService (no lifespan is run in these tests)."""

    def __init__(
        self, response: RecommendResponse | None = None, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error

    def recommend(self, text: str) -> RecommendResponse:
        if self._error:
            raise self._error
        assert self._response is not None
        return self._response


@pytest.fixture
async def client():
    # No `async with lifespan(app)` here on purpose: the route only needs
    # app.state.recommendation_service, and skipping startup keeps these
    # tests fast and independent of a real database/OpenAI key.
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


async def test_recommend_recipe_returns_markdown_and_decision(client: AsyncClient) -> None:
    app.state.recommendation_service = StubService(response=_FAKE_RESPONSE)

    response = await client.post("/recommend_recipe", json={"text": "I have chicken and soy sauce"})

    assert response.status_code == 200
    body = response.json()
    assert body["recipe"] == _FAKE_RESPONSE.recipe
    assert body["decision"]["selected_recipe"] == "Quick Chicken Stir-Fry"
    assert body["decision"]["missing_ingredients"] == ["mixed vegetables", "garlic"]


async def test_recommend_recipe_rejects_empty_text(client: AsyncClient) -> None:
    response = await client.post("/recommend_recipe", json={"text": "   "})

    assert response.status_code == 422


async def test_recommend_recipe_rejects_missing_field(client: AsyncClient) -> None:
    response = await client.post("/recommend_recipe", json={})

    assert response.status_code == 422


async def test_recommend_recipe_maps_retrieval_error_to_502_without_leaking_details(
    client: AsyncClient,
) -> None:
    app.state.recommendation_service = StubService(
        error=RetrievalError("pgvector connection refused")
    )

    response = await client.post("/recommend_recipe", json={"text": "I have chicken"})

    assert response.status_code == 502
    assert "pgvector" not in response.text
    assert "connection refused" not in response.text


async def test_recommend_recipe_maps_generation_error_to_502_without_leaking_details(
    client: AsyncClient,
) -> None:
    app.state.recommendation_service = StubService(
        error=GenerationError("invalid api key for openai")
    )

    response = await client.post("/recommend_recipe", json={"text": "I have chicken"})

    assert response.status_code == 502
    assert "api key" not in response.text
