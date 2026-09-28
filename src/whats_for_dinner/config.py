"""Application configuration loaded from environment variables / .env.

Kept as a single settings object so every other module has one obvious
place to get configuration from, instead of reading os.environ directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Small, explicit, easy to extend when a new model is adopted - same pattern
# as PANTRY_STAPLES in prompts.py. A model that isn't listed here is simply
# not cross-checked (see _check_embedding_dimension below), not rejected.
_KNOWN_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class Settings(BaseSettings):
    openai_api_key: str
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    # Must match the output dimension of openai_embedding_model - pgvector
    # needs a fixed vector size when the table/column is created. Checked
    # against _KNOWN_EMBEDDING_DIMENSIONS below rather than trusted blindly,
    # since a mismatch here would otherwise only surface later as a confusing
    # dimension error from pgvector during ingestion.
    embedding_dimension: int = 1536

    database_url: str
    recipe_data_path: Path = Path("data/recipes")
    retrieval_top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @model_validator(mode="after")
    def _check_embedding_dimension(self) -> Self:
        expected_dimension = _KNOWN_EMBEDDING_DIMENSIONS.get(self.openai_embedding_model)
        if expected_dimension is not None and expected_dimension != self.embedding_dimension:
            raise ValueError(
                f"EMBEDDING_DIMENSION={self.embedding_dimension} does not match "
                f"OPENAI_EMBEDDING_MODEL={self.openai_embedding_model!r}, which produces "
                f"{expected_dimension}-dimensional vectors. Update .env."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so .env is parsed once per process."""
    # Pydantic's dataclass_transform makes pyright synthesize an __init__
    # that requires every field; BaseSettings actually takes **values at
    # runtime and sources required fields from the environment/.env instead.
    return Settings()  # pyright: ignore[reportCallIssue]
