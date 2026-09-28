"""Load recipe .txt files into Haystack Documents.

One recipe file -> one Document. Recipes in this corpus are short cohesive
units (title + ingredients + instructions); splitting them into chunks would
separate ingredients from the instructions that use them, which is the one
thing retrieval most needs to keep together.
"""

import hashlib
import logging
from pathlib import Path

from haystack import Document

logger = logging.getLogger(__name__)


def load_recipes(data_path: Path) -> list[Document]:
    """Read every .txt recipe under data_path into one Document each.

    Files are processed in sorted filename order so ingestion behaves the
    same way on every run. Empty files are skipped rather than failing the
    whole batch, since one bad file shouldn't block startup.
    """
    documents: list[Document] = []
    for file_path in sorted(data_path.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            logger.warning("Skipping empty recipe file: %s", file_path.name)
            continue

        documents.append(
            Document(
                id=_compute_document_id(file_path.name, content),
                content=content,
                meta={"title": _extract_title(content), "source": file_path.name},
            )
        )
    return documents


def _extract_title(content: str) -> str:
    # Recipe files consistently lead with the title, sometimes after a blank line.
    for line in content.splitlines():
        if line.strip():
            return line.strip()
    return "Untitled Recipe"


def _compute_document_id(filename: str, content: str) -> str:
    """Deterministic ID from filename + normalized content.

    Restarting the app re-runs ingestion against the same files, so the ID
    must be stable across runs (idempotency) but change if a recipe's text
    is edited (so the corpus can't go stale silently).
    """
    normalized_content = " ".join(content.split())
    fingerprint = f"{filename}:{normalized_content}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
