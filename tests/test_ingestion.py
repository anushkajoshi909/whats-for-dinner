from pathlib import Path

from haystack import Document

from whats_for_dinner.ingestion import ingest_recipes
from whats_for_dinner.recipes import load_recipes


class FakeDocumentStore:
    """Minimal in-memory stand-in for PgvectorDocumentStore's methods used by ingestion."""

    def __init__(self, existing: list[Document] | None = None) -> None:
        self._documents_by_id: dict[str, Document] = {doc.id: doc for doc in existing or []}
        self.deleted_ids: list[str] = []

    def filter_documents(self, filters: dict[str, object] | None = None) -> list[Document]:
        return list(self._documents_by_id.values())

    def write_documents(self, documents: list[Document], policy: object = None) -> int:
        for document in documents:
            self._documents_by_id[document.id] = document
        return len(documents)

    def delete_documents(self, document_ids: list[str]) -> None:
        self.deleted_ids.extend(document_ids)
        for document_id in document_ids:
            self._documents_by_id.pop(document_id, None)


class FakeDocumentEmbedder:
    """Stands in for OpenAIDocumentEmbedder so tests never call the real API."""

    def __init__(self) -> None:
        self.embedded_batches: list[list[Document]] = []

    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        self.embedded_batches.append(documents)
        for document in documents:
            document.embedding = [0.0]
        return {"documents": documents}


def _recipe_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    for filename, content in files.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")
    return tmp_path


def test_ingest_recipes_embeds_and_stores_everything_on_first_run(tmp_path: Path) -> None:
    data_path = _recipe_dir(
        tmp_path, {"01.txt": "Recipe One\ncontent", "02.txt": "Recipe Two\ncontent"}
    )
    store = FakeDocumentStore()
    embedder = FakeDocumentEmbedder()

    ingested_count = ingest_recipes(store, embedder, data_path)

    assert ingested_count == 2
    assert len(embedder.embedded_batches[0]) == 2
    assert len(store.filter_documents()) == 2


def test_ingest_recipes_is_idempotent_on_repeated_calls(tmp_path: Path) -> None:
    data_path = _recipe_dir(tmp_path, {"01.txt": "Recipe One\ncontent"})
    store = FakeDocumentStore()
    embedder = FakeDocumentEmbedder()
    ingest_recipes(store, embedder, data_path)

    second_run_count = ingest_recipes(store, embedder, data_path)

    assert second_run_count == 0
    assert len(embedder.embedded_batches) == 1  # embedder was not called again
    assert len(store.filter_documents()) == 1  # no duplicate rows


def test_ingest_recipes_only_embeds_documents_not_already_stored(tmp_path: Path) -> None:
    data_path = _recipe_dir(
        tmp_path, {"01.txt": "Recipe One\ncontent", "02.txt": "Recipe Two\ncontent"}
    )
    # Pre-seed the store with 01.txt's real ID so only 02.txt should need embedding.
    existing_doc = next(doc for doc in load_recipes(data_path) if doc.meta["source"] == "01.txt")
    store = FakeDocumentStore(existing=[existing_doc])
    embedder = FakeDocumentEmbedder()

    ingested_count = ingest_recipes(store, embedder, data_path)

    assert ingested_count == 1
    assert embedder.embedded_batches[0][0].meta["source"] == "02.txt"


def test_ingest_recipes_removes_stale_version_when_recipe_content_changes(tmp_path: Path) -> None:
    data_path = _recipe_dir(tmp_path, {"01.txt": "Recipe One\nupdated content"})
    stale_doc = Document(
        id="stale-id-from-old-content",
        content="old content",
        meta={"source": "01.txt", "title": "Recipe One"},
    )
    store = FakeDocumentStore(existing=[stale_doc])
    embedder = FakeDocumentEmbedder()

    ingest_recipes(store, embedder, data_path)

    assert store.deleted_ids == ["stale-id-from-old-content"]
    remaining = store.filter_documents()
    assert len(remaining) == 1
    assert remaining[0].id != "stale-id-from-old-content"
