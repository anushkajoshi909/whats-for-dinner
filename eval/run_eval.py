"""Evaluation runner: retrieval + generation metrics against dataset.jsonl.

This is a small script, not a platform (PIPELINE.md section 22): it runs the real
pipeline (real DB, real OpenAI calls) against a human-annotated query set and
prints retrieval metrics separately from generation metrics, plus a
per-query failure-stage classification. Run it explicitly and sparingly -
every row makes at least one embedding call and one GPT-4o call.

Usage (from the project root, with the db running and .env populated):
    uv run python eval/run_eval.py

Each run overwrites results.json alongside this script with the same numbers
as structured data, so a later run (e.g. after a prompt change) can be
compared against it without re-reading terminal scrollback.
"""

import statistics
from pathlib import Path

from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from pydantic import BaseModel

from whats_for_dinner.config import get_settings
from whats_for_dinner.generation import generate_recommendation
from whats_for_dinner.models import RecommendationDecision, RetrievedCandidate
from whats_for_dinner.prompts import build_recommendation_messages
from whats_for_dinner.recipes import load_recipes
from whats_for_dinner.retrieval import retrieve_candidates

_DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
_RESULTS_PATH = Path(__file__).parent / "results.json"


class EvalRecord(BaseModel):
    """One human-annotated row. Pydantic validates the fixture on load,
    catching a malformed dataset row before it reaches the live pipeline."""

    query: str
    relevant_recipe_ids: list[str]
    preferred_recipe_id: str | None
    expected_matched_ingredients: list[str]
    expected_missing_ingredients: list[str]
    expected_constraint_conflicts: list[str]
    expected_is_reasonable_match: bool


class QueryResult:
    def __init__(
        self,
        record: EvalRecord,
        candidates: list[RetrievedCandidate],
        decision: RecommendationDecision,
        failure_stage: str,
    ) -> None:
        self.record = record
        self.candidates = candidates
        self.decision = decision
        self.failure_stage = failure_stage


class RetrievalMetrics(BaseModel):
    """Candidate-generation quality - see PIPELINE.md section 19."""

    top_k: int
    recall_at_k: float
    precision_at_k: float
    hit_rate_at_k: float
    mrr: float


class GenerationMetrics(BaseModel):
    """Decision quality, given retrieval succeeded - see PIPELINE.md section 20."""

    selection_accuracy: float
    is_reasonable_match_accuracy: float
    constraint_conflicts_precision: float
    constraint_conflicts_recall: float
    matched_ingredients_precision: float
    matched_ingredients_recall: float
    missing_ingredients_precision: float
    missing_ingredients_recall: float


class QueryReportEntry(BaseModel):
    """One row of the per-query breakdown, saved alongside the aggregate metrics
    so a regression can be traced back to the specific query that caused it."""

    query: str
    selected_recipe_title: str
    is_reasonable_match: bool
    failure_stage: str


class EvalReport(BaseModel):
    """The full contents of results.json."""

    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    results: list[QueryReportEntry]


def _load_dataset() -> list[EvalRecord]:
    with _DATASET_PATH.open(encoding="utf-8") as f:
        return [EvalRecord.model_validate_json(line) for line in f if line.strip()]


def _short_id_to_doc_id(data_path: Path) -> dict[str, str]:
    """Map a fixture's human-friendly "01" reference to the real content-hash document ID."""
    return {doc.meta["source"].removesuffix(".txt"): doc.id for doc in load_recipes(data_path)}


def _ingredient_prf(expected: list[str], actual: list[str]) -> tuple[float, float]:
    """Case-insensitive set precision/recall. Returns (1.0, 1.0) when both sides are empty."""
    expected_set = {item.lower() for item in expected}
    actual_set = {item.lower() for item in actual}
    if not expected_set and not actual_set:
        return 1.0, 1.0
    if not actual_set:
        return 0.0, 0.0
    if not expected_set:
        return 0.0, 1.0
    true_positives = len(expected_set & actual_set)
    precision = true_positives / len(actual_set)
    recall = true_positives / len(expected_set)
    return precision, recall


def _classify_failure(
    record: EvalRecord,
    candidate_doc_ids: list[str],
    preferred_doc_id: str | None,
    decision: RecommendationDecision,
) -> str:
    """One label per PIPELINE.md section 21's taxonomy, in priority order."""
    if preferred_doc_id is not None and preferred_doc_id not in candidate_doc_ids:
        return "retrieval failure"
    if record.expected_is_reasonable_match != decision.is_reasonable_match:
        return "selection failure"
    if preferred_doc_id is not None and decision.selected_recipe_id != preferred_doc_id:
        return "selection failure"
    constraint_precision, constraint_recall = _ingredient_prf(
        record.expected_constraint_conflicts, decision.constraint_conflicts
    )
    if constraint_precision < 1.0 or constraint_recall < 1.0:
        return "constraint failure"
    _, matched_recall = _ingredient_prf(
        record.expected_matched_ingredients, decision.matched_ingredients
    )
    _, missing_recall = _ingredient_prf(
        record.expected_missing_ingredients, decision.missing_ingredients
    )
    if matched_recall < 0.5 or missing_recall < 0.5:
        return "ingredient-accounting failure"
    return "pass"


def _evaluate_one(
    record: EvalRecord,
    text_embedder: OpenAITextEmbedder,
    retriever: PgvectorEmbeddingRetriever,
    chat_generator: OpenAIChatGenerator,
    id_map: dict[str, str],
) -> QueryResult:
    candidates = retrieve_candidates(text_embedder, retriever, record.query)
    messages = build_recommendation_messages(record.query, candidates)
    decision = generate_recommendation(chat_generator, messages)

    candidate_doc_ids = [c.recipe_id for c in candidates]
    preferred_doc_id = id_map[record.preferred_recipe_id] if record.preferred_recipe_id else None
    failure_stage = _classify_failure(record, candidate_doc_ids, preferred_doc_id, decision)
    return QueryResult(
        record=record, candidates=candidates, decision=decision, failure_stage=failure_stage
    )


def _compute_retrieval_metrics(
    results: list[QueryResult], id_map: dict[str, str], top_k: int
) -> RetrievalMetrics:
    recalls: list[float] = []
    precisions: list[float] = []
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    for result in results:
        relevant_doc_ids = {id_map[short_id] for short_id in result.record.relevant_recipe_ids}
        if not relevant_doc_ids:
            continue  # undefined for the no-good-match query; judged on is_reasonable_match instead
        retrieved_doc_ids = [c.recipe_id for c in result.candidates[:top_k]]
        hit_positions = [
            i for i, doc_id in enumerate(retrieved_doc_ids, start=1) if doc_id in relevant_doc_ids
        ]

        recalls.append(len(hit_positions) / len(relevant_doc_ids))
        precisions.append(len(hit_positions) / top_k)
        hits.append(1.0 if hit_positions else 0.0)
        reciprocal_ranks.append(1.0 / hit_positions[0] if hit_positions else 0.0)

    return RetrievalMetrics(
        top_k=top_k,
        recall_at_k=statistics.mean(recalls),
        precision_at_k=statistics.mean(precisions),
        hit_rate_at_k=statistics.mean(hits),
        mrr=statistics.mean(reciprocal_ranks),
    )


def _compute_generation_metrics(
    results: list[QueryResult], id_map: dict[str, str]
) -> GenerationMetrics:
    selection_correct: list[float] = []
    reasonable_match_correct: list[float] = []
    constraint_precisions: list[float] = []
    constraint_recalls: list[float] = []
    matched_precisions: list[float] = []
    matched_recalls: list[float] = []
    missing_precisions: list[float] = []
    missing_recalls: list[float] = []

    for result in results:
        record, decision = result.record, result.decision
        preferred_doc_id = (
            id_map[record.preferred_recipe_id] if record.preferred_recipe_id else None
        )
        selection_correct.append(
            1.0
            if preferred_doc_id is None or decision.selected_recipe_id == preferred_doc_id
            else 0.0
        )
        reasonable_match_correct.append(
            1.0 if decision.is_reasonable_match == record.expected_is_reasonable_match else 0.0
        )
        # Precision/recall over the actual conflicting items, not just whether *some*
        # conflict was reported - a set comparison catches both a missed conflict and
        # a hallucinated one, where a bool comparison would only catch the former.
        cp, cr = _ingredient_prf(
            record.expected_constraint_conflicts, decision.constraint_conflicts
        )
        constraint_precisions.append(cp)
        constraint_recalls.append(cr)
        mp, mr = _ingredient_prf(record.expected_matched_ingredients, decision.matched_ingredients)
        xp, xr = _ingredient_prf(record.expected_missing_ingredients, decision.missing_ingredients)
        matched_precisions.append(mp)
        matched_recalls.append(mr)
        missing_precisions.append(xp)
        missing_recalls.append(xr)

    return GenerationMetrics(
        selection_accuracy=statistics.mean(selection_correct),
        is_reasonable_match_accuracy=statistics.mean(reasonable_match_correct),
        constraint_conflicts_precision=statistics.mean(constraint_precisions),
        constraint_conflicts_recall=statistics.mean(constraint_recalls),
        matched_ingredients_precision=statistics.mean(matched_precisions),
        matched_ingredients_recall=statistics.mean(matched_recalls),
        missing_ingredients_precision=statistics.mean(missing_precisions),
        missing_ingredients_recall=statistics.mean(missing_recalls),
    )


def _build_report(results: list[QueryResult], id_map: dict[str, str], top_k: int) -> EvalReport:
    return EvalReport(
        retrieval=_compute_retrieval_metrics(results, id_map, top_k),
        generation=_compute_generation_metrics(results, id_map),
        results=[
            QueryReportEntry(
                query=result.record.query,
                selected_recipe_title=result.decision.selected_recipe_title,
                is_reasonable_match=result.decision.is_reasonable_match,
                failure_stage=result.failure_stage,
            )
            for result in results
        ],
    )


def _print_report(report: EvalReport) -> None:
    r, g = report.retrieval, report.generation
    print("\n--- Retrieval metrics (candidate generation quality) ---")
    print(f"Recall@{r.top_k}:    {r.recall_at_k:.2f}")
    print(f"Precision@{r.top_k}: {r.precision_at_k:.2f}")
    print(f"Hit rate@{r.top_k}:  {r.hit_rate_at_k:.2f}")
    print(f"MRR:          {r.mrr:.2f}")

    print("\n--- Generation / decision metrics ---")
    print(f"Selection accuracy:           {g.selection_accuracy:.2f}")
    print(f"is_reasonable_match accuracy: {g.is_reasonable_match_accuracy:.2f}")
    constraint_p, constraint_r = g.constraint_conflicts_precision, g.constraint_conflicts_recall
    print(f"Constraint conflicts P/R:     {constraint_p:.2f} / {constraint_r:.2f}")
    matched_p, matched_r = g.matched_ingredients_precision, g.matched_ingredients_recall
    missing_p, missing_r = g.missing_ingredients_precision, g.missing_ingredients_recall
    print(f"Matched ingredients  P/R:     {matched_p:.2f} / {matched_r:.2f}")
    print(f"Missing ingredients  P/R:     {missing_p:.2f} / {missing_r:.2f}")

    print("\n--- Per-query result ---")
    for entry in report.results:
        print(f"[{entry.failure_stage:<28}] {entry.query!r} -> {entry.selected_recipe_title!r}")


def main() -> None:
    settings = get_settings()
    openai_api_key = Secret.from_token(settings.openai_api_key)

    document_store = PgvectorDocumentStore(
        connection_string=Secret.from_token(settings.database_url),
        table_name="recipes",
        embedding_dimension=settings.embedding_dimension,
    )
    text_embedder = OpenAITextEmbedder(
        api_key=openai_api_key, model=settings.openai_embedding_model
    )
    retriever = PgvectorEmbeddingRetriever(
        document_store=document_store, top_k=settings.retrieval_top_k
    )
    chat_generator = OpenAIChatGenerator(api_key=openai_api_key, model=settings.openai_chat_model)

    id_map = _short_id_to_doc_id(settings.recipe_data_path)
    dataset = _load_dataset()

    results = [
        _evaluate_one(record, text_embedder, retriever, chat_generator, id_map)
        for record in dataset
    ]

    report = _build_report(results, id_map, settings.retrieval_top_k)
    _print_report(report)

    _RESULTS_PATH.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
