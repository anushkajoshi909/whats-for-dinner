"""Idempotent recipe ingestion into the pgvector document store.

Runs at startup because the corpus is small (PIPELINE.md sections 5 and 7). Idempotency
comes from the deterministic document IDs in recipes.py: this module only
embeds and writes documents that aren't already present, and cleans up any
previous version of a recipe whose content changed.
"""

import logging
from pathlib import Path
from typing import Protocol

from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from whats_for_dinner.recipes import load_recipes

logger = logging.getLogger(__name__)


class RecipeDocumentStore(Protocol):
    """The PgvectorDocumentStore methods this module depends on."""

    def filter_documents(self, filters: dict[str, object] | None = None) -> list[Document]: ...
    def write_documents(self, documents: list[Document], policy: DuplicatePolicy) -> int: ...
    def delete_documents(self, document_ids: list[str]) -> None: ...


class DocumentEmbedder(Protocol):
    """The one OpenAIDocumentEmbedder method this module depends on."""

    def run(self, documents: list[Document]) -> dict[str, list[Document]]: ...


def ingest_recipes(
    document_store: RecipeDocumentStore,
    embedder: DocumentEmbedder,
    data_path: Path,
) -> int:
    """Ensure every recipe under data_path has an up-to-date embedded Document.

    Returns the number of documents newly embedded and written. Safe to call
    on every application startup.
    """
    documents = load_recipes(data_path)
    existing_documents = document_store.filter_documents()

    _remove_stale_versions(document_store, documents, existing_documents)

    existing_ids = {doc.id for doc in existing_documents}
    new_documents = [doc for doc in documents if doc.id not in existing_ids]
    if not new_documents:
        logger.info("Recipe ingestion: nothing new (%d recipes already indexed)", len(documents))
        return 0

    embedded_documents = embedder.run(documents=new_documents)["documents"]
    document_store.write_documents(embedded_documents, policy=DuplicatePolicy.SKIP)
    logger.info("Recipe ingestion: embedded and stored %d new recipe(s)", len(embedded_documents))
    return len(embedded_documents)


def _remove_stale_versions(
    document_store: RecipeDocumentStore,
    current_documents: list[Document],
    existing_documents: list[Document],
) -> None:
    """Delete a previously stored recipe version whose content has since changed.

    Since the document ID is derived from filename + content, editing a
    recipe file produces a new ID for the same source file; without this the
    old embedding would linger in the store forever.
    """
    current_id_by_source = {doc.meta["source"]: doc.id for doc in current_documents}
    stale_ids = [
        doc.id
        for doc in existing_documents
        if doc.meta.get("source") in current_id_by_source
        and doc.id != current_id_by_source[doc.meta["source"]]
    ]
    if stale_ids:
        document_store.delete_documents(stale_ids)
        logger.info("Recipe ingestion: removed %d stale recipe version(s)", len(stale_ids))
