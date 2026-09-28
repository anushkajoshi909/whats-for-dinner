# What's for Dinner

A small RAG application that recommends one recipe from a fixed corpus based on a free-text
description of the ingredients and constraints a user has. Built with FastAPI, Haystack 2.x,
PostgreSQL/pgvector, and GPT-4o.

> Retrieval finds plausible candidate recipes; it does not make the final recommendation
> decision. GPT-4o compares the retrieved candidates against the user's actual ingredients and
> constraints and produces an explainable, structured decision.

## Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Using the API](#using-the-api)
- [How startup ingestion works](#how-startup-ingestion-works)
- [The structured recommendation decision](#the-structured-recommendation-decision)
- [Pantry-staple policy](#pantry-staple-policy)
- [Design decisions & trade-offs](#design-decisions--trade-offs)
- [Limitations](#limitations)
- [Tests](#tests)
- [Evaluation strategy](#evaluation-strategy)
- [Image input (bonus, not implemented)](#image-input-bonus-not-implemented)
- [Future improvements](#future-improvements)

## Architecture

**Startup / ingestion** (runs once, at process startup):

```
data/recipes/*.txt -> recipe loader -> one Document per recipe (deterministic ID)
                                     -> OpenAI document embedder
                                     -> PostgreSQL/pgvector
```

**Request path**:

```
POST /recommend_recipe
    -> Pydantic request validation (reject empty/whitespace text)
    -> RecommendationService
         -> OpenAI query embedder
         -> pgvector top-k retrieval (candidates + rank + similarity score)
         -> prompt builder (rules + pantry-staple policy + candidates)
         -> GPT-4o, strict JSON-schema structured output
         -> RecommendationDecision (validated Pydantic model)
    -> structured trace logged (request id, candidates, decision, latency)
    -> RecommendResponse (Markdown + a decision summary)
```

### Project layout

```
src/whats_for_dinner/
  config.py       Pydantic Settings (env vars / .env)
  errors.py       RetrievalError, GenerationError - the only two domain exceptions
  models.py       Request/response models + RecommendationDecision + trace model
  recipes.py      Load .txt files -> Documents, deterministic IDs
  ingestion.py    Idempotent embed-and-store
  retrieval.py    Query embedding + pgvector retrieval (+ Protocol interfaces)
  prompts.py      The recommendation prompt and the pantry-staple list (one place, easy to edit)
  generation.py   GPT-4o call with strict structured output + schema validation
  service.py      RecommendationService - the top-down business flow for one request
  main.py         FastAPI app, component wiring, startup ingestion, the one thin route
  custom_components.py   Supplied helper for the image bonus (unused - see below)
tests/            pytest, one file per module, external calls mocked with small fakes
eval/             Human-annotated evaluation fixture + a runner script (see below)
```

`data/recipes/` (the 20 supplied `.txt` files) is the corpus actually used at runtime;
`data.zip` is kept as the original supplied archive and isn't read by the application.

Everything is flat inside `src/whats_for_dinner/` on purpose - the corpus and the pipeline are
small enough that splitting into sub-packages would add navigation overhead without a real
benefit.

## Setup

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- A Docker-compatible runtime (Docker Desktop, or Colima + the `docker` CLI)
- An OpenAI API key with access to `gpt-4o` and an embedding model

### 1. Start PostgreSQL/pgvector

```bash
docker compose up -d
```

This starts the supplied `ankane/pgvector` image on `localhost:5432` with the credentials in
`docker-compose.yml`. No other datastore is introduced.

> The supplied compose file has no named volume, so `docker compose down` (which removes the
> container) also wipes the database - `docker compose stop`/`start` preserves it. This is a
> non-issue in practice: ingestion is idempotent and cheap (20 recipes), so the app fully
> self-heals - including recreating the `vector` extension and the table - on next startup.

### 2. Configure environment

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | required, never logged/printed | - |
| `OPENAI_CHAT_MODEL` | generation model | `gpt-4o` |
| `OPENAI_EMBEDDING_MODEL` | embedding model | `text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | must match the embedding model's output size | `1536` |
| `DATABASE_URL` | pgvector connection string | matches `docker-compose.yml` |
| `RECIPE_DATA_PATH` | folder of recipe `.txt` files | `data/recipes` |
| `RETRIEVAL_TOP_K` | candidates retrieved before GPT-4o reasoning | `5` |

### 3. Install dependencies

```bash
uv sync
```

This creates `.venv` and installs the pinned dependencies plus a `dev` group (pytest, ruff,
pyright, anyio, httpx). `pgvector-haystack==3.4.1` was added to `pyproject.toml` - it's the
Postgres/pgvector integration package for Haystack, and isn't part of `haystack-ai` itself (see
[Design decisions](#design-decisions--trade-offs)).

### 4. Run the app

```bash
uv run uvicorn whats_for_dinner.main:app --reload
```

On startup you'll see recipe ingestion run (embeds anything not already in the database - see
below), then `Application startup complete.` The API is at `http://127.0.0.1:8000`.

## Using the API

### Straightforward match

```bash
curl -s -X POST http://127.0.0.1:8000/recommend_recipe \
  -H 'Content-Type: application/json' \
  -d '{"text": "I have chicken, broccoli and soy sauce"}'
```

```json
{
  "recipe": "# Quick Chicken Stir-Fry\n\n**Ingredients:**\n\n- 2 chicken breasts, diced\n...",
  "decision": {
    "selected_recipe": "Quick Chicken Stir-Fry",
    "matched_ingredients": ["chicken", "soy sauce"],
    "missing_ingredients": ["mixed vegetables", "garlic"],
    "assumed_pantry_staples": ["vegetable oil"],
    "constraint_conflicts": [],
    "is_reasonable_match": true
  }
}
```

### Explicit constraint / no good candidate

```bash
curl -s -X POST http://127.0.0.1:8000/recommend_recipe \
  -H 'Content-Type: application/json' \
  -d '{"text": "I have pasta, tomatoes and basil. No cheese please, I am dairy-free."}'
```

Returns `is_reasonable_match: false` with `constraint_conflicts: ["cheese"]` and a Markdown
explanation of what would need to change - it does not silently pick a conflicting recipe or
invent a substitute recipe that isn't in the corpus.

### Empty input

```bash
curl -s -X POST http://127.0.0.1:8000/recommend_recipe -d '{"text": "   "}'
# -> 422, {"detail": [...]}
```

### Response contract

```python
class RecommendResponse(BaseModel):
    recipe: str            # Markdown recommendation
    decision: DecisionSummary
```

The full structured decision (`RecommendationDecision`) - including `decision_reason` and
`selected_recipe_id` - is kept internal (logged per request) rather than added to the public
response, per the challenge's request to keep the API contract simple; the summary already
carries matched/missing ingredients, pantry assumptions, constraint conflicts and the
reasonable-match flag.

## How startup ingestion works

Each recipe file becomes exactly one `Document` (recipes are short and cohesive - splitting
ingredients from instructions would only hurt retrieval). The document ID is
`sha256(filename + normalized_content)`:

- **Idempotent**: re-running ingestion (e.g. every app restart) computes the same ID for
  unchanged files, so `ingest_recipes` only embeds documents whose ID isn't already in the
  store. Restarting the app makes zero embedding calls once the corpus is indexed.
- **Self-healing on edits**: if a recipe file's content changes, its ID changes; the old
  version (matched by `meta.source`) is deleted so the store never accumulates stale
  duplicates of the same file.

Verified manually: two consecutive startups against the supplied 20 recipes ingest 20 the
first time and 0 the second, with the row count in `pgvector` unchanged.

## The structured recommendation decision

`OpenAIChatGenerator.run` in the pinned Haystack version (`2.12.0`) has no `response_format`
parameter - that landed in later Haystack releases. Structured output is requested the way the
pinned `openai==1.75.0` client supports it directly: a strict JSON-schema `response_format`
passed through `generation_kwargs`, which `run` forwards unchanged into
`openai.chat.completions.create(**api_args)`. The reply is then parsed with
`RecommendationDecision.model_validate_json(...)`, so a malformed reply fails loudly as a
`GenerationError` instead of silently producing bad data. This was verified directly against a
live GPT-4o call before being wired into the app (see `generation.py`'s module docstring).

```python
class RecommendationDecision(BaseModel):
    selected_recipe_id: str
    selected_recipe_title: str
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    assumed_pantry_staples: list[str]
    constraint_conflicts: list[str]
    decision_reason: str
    is_reasonable_match: bool
    markdown: str
```

Every request logs one structured record (`service.py`) with: request ID, the candidate IDs/
titles/ranks/scores that were retrieved, the full decision above, the model/embedding names
used, and latency - enough to answer "was this a retrieval problem or a generation problem?"
without ever logging the API key or database credentials. As a safety net, `generation.py` also
strips any ingredient the model lists in *both* `missing_ingredients` and
`assumed_pantry_staples` (observed once during manual testing) so the documented invariant -
each ingredient in exactly one category - holds even when the model doesn't follow the prompt's
instruction perfectly.

## Pantry-staple policy

A small, explicit, easy-to-change list in `prompts.py`:

```python
PANTRY_STAPLES: list[str] = ["salt", "pepper", "water", "cooking oil"]
```

A recipe ingredient the user didn't mention is `assumed_pantry_staples` only if it's on this
list; everything else the user didn't mention is a genuine `missing_ingredient`. The model is
explicitly instructed not to invent pantry assumptions beyond this list.

## Design decisions & trade-offs

- **One document per recipe.** Recipes are short cohesive units; chunking would separate
  ingredients from the instructions that use them.
- **pgvector, no other vector store.** Supplied by the challenge; adding a second datastore
  would be unjustified infrastructure for a 20-recipe PoC.
- **`pgvector-haystack` as an added dependency.** Haystack 2.12 ships no Postgres document
  store; the integration lives in a separate package. Version `3.4.1` was chosen because it's
  the newest release that still declares `haystack-ai>=2.11.0` (matching the pinned `2.12.0`) -
  versions `6.x` require `haystack-ai>=2.22.0` and don't install alongside the pinned version.
- **Semantic retrieval is candidate generation, not the final decision** (PIPELINE.md's central
  design principle). No hard similarity threshold is applied without evaluation evidence -
  `top_k` is the only retrieval lever.
- **GPT-4o performs the final candidate reasoning**, comparing the user's literal ingredients
  against each candidate's real ingredient list, because vector similarity can't tell whether
  the user actually has what a recipe needs.
- **Structured decision over free-form Markdown.** `matched_ingredients` / `missing_ingredients`
  / `assumed_pantry_staples` are explicit fields, not something to reverse-engineer from prose,
  so the system is debuggable and evaluable.
- **Quantities stay natural language.** The full user request is preserved and embedded/passed
  to GPT-4o as-is; no unit/measurement normalization subsystem was built.
- **Startup ingestion, not a separate job.** Justified by the challenge and the corpus size (20
  recipes); a dedicated indexing job is the documented scaling path.
- **Ingestion is idempotent** via content-derived deterministic IDs (see above).
- **No repository-layer wrapping around `PgvectorDocumentStore`.** It already is the storage
  abstraction; wrapping it again would add indirection with no behavioral benefit. Where test
  isolation was needed (ingestion, retrieval, generation), small `typing.Protocol` interfaces
  describe only the one or two methods each module actually calls - a zero-runtime-cost typing
  device, not another architectural layer. Real Haystack components satisfy them structurally
  with no changes at the call sites.
- **No SQLModel / custom tables.** `PgvectorDocumentStore` owns its own table schema; adding
  SQLModel models for a table this app never queries directly would be pure ceremony.
- **Sync Haystack calls, async API boundary.** In the pinned Haystack version,
  `OpenAIDocumentEmbedder` and `OpenAITextEmbedder` have no `run_async`, so the whole pipeline
  is sync internally; the one blocking call per request is offloaded with
  `asyncio.to_thread` at the FastAPI route rather than forcing a partially-async pipeline for
  components that don't support it.
- **Human-annotated evaluation set**, not the evaluated model's own output, as ground truth (see
  below).

## Limitations

- The 20-recipe corpus is tiny; retrieval quality at this scale doesn't generalize to a larger,
  noisier catalog.
- Ingredient matching is exact-string (case-insensitive) inside the LLM's own reasoning - there's
  no synonym/ontology layer, so "veggies" vs. "mixed vegetables" relies entirely on GPT-4o's
  judgment, not a deterministic matcher.
- `missing_ingredients` recall isn't perfect - see [Evaluation strategy](#evaluation-strategy);
  the model sometimes under-reports a genuinely missing ingredient it considers minor (e.g. the
  cheese folded into a pesto sauce, or a spice blend). This is a real, measured generation
  weakness, not a retrieval problem.
- **Recipe selection is not fully deterministic.** Two consecutive evaluation runs against the
  identical prompt/candidates produced different selection accuracy (1.00 vs. 0.80 - see
  [Evaluation strategy](#evaluation-strategy)), including one run where a clearly worse candidate
  was picked over a clearly better one already in the retrieved set. A single successful manual
  test, or even a single eval run, is not sufficient evidence this system behaves correctly -
  it should be run multiple times per change before trusting a metric movement as real.
- `custom_components.py` (the supplied image-ingredient-extraction helper) has pre-existing
  `pyright` errors from the challenge's own reference code; it isn't imported or used anywhere
  in the text pipeline, and wasn't modified.
- No auth, rate limiting, or production observability - explicitly out of scope for a PoC.

## Tests

```bash
uv run pytest       # 26 tests, all OpenAI/DB calls replaced with small fakes, <1s
uv run ruff check .
uv run pyright
```

Coverage, by file:

- `test_recipes.py` - title extraction, deterministic IDs (stable across reruns, change when
  content changes, differ across filenames, insensitive to whitespace reformatting).
- `test_ingestion.py` - first-run embeds everything; second run embeds nothing (idempotency);
  only genuinely new documents are embedded; stale versions are deleted on content change.
- `test_generation.py` - valid replies parse; the strict JSON-schema `response_format` is
  actually sent; malformed JSON, a missing required field, and an API failure all raise
  `GenerationError`; the pantry-staple/missing-ingredient dedup safety net works.
- `test_service.py` - the service's own orchestration (retrieve -> prompt -> generate -> map to
  response), and that `RetrievalError`/`GenerationError` propagate correctly.
- `test_api.py` - the public response contract, empty/missing-field input -> 422, and that
  `RetrievalError`/`GenerationError` map to 502 without leaking the underlying exception text
  (verified by asserting the raw error strings are *not* present in the response body).

## Evaluation strategy

Retrieval and generation are evaluated independently, against a small **human-annotated**
fixture (`eval/dataset.jsonl`, 10 queries) - never against the evaluated model's own output. Run
it (costs real OpenAI calls, run sparingly):

```bash
uv run python eval/run_eval.py
```

Each run prints the report and overwrites `eval/results.json` next to it (aggregate metrics
plus a per-query breakdown, as structured data) - so one run can be diffed against a later one
after a prompt or model change, instead of relying on terminal scrollback. It isn't committed
(`.gitignore`) since GPT-4o's output isn't fully deterministic, so its contents differ run to
run - see below.

Each record has `query`, `relevant_recipe_ids`, `preferred_recipe_id`, expected
matched/missing ingredients, whether a constraint conflict is expected, and whether a
reasonable match is expected to exist at all.

**Retrieval metrics** (candidate generation quality - "did retrieval hand generation the right
recipe at all?"): Recall@K, Precision@K, Hit rate@K, MRR.

**Generation metrics** (decision quality, given retrieval succeeded): selection accuracy,
`is_reasonable_match` accuracy, constraint adherence, and matched/missing ingredient
precision/recall (case-insensitive set comparison against the human labels).

**Failure taxonomy** per query (`_classify_failure` in `eval/run_eval.py`), in priority order:
`retrieval failure` (preferred recipe never reached the candidate set) -> `selection failure`
(wrong candidate chosen, or `is_reasonable_match` disagrees with the label) -> `constraint
failure` -> `ingredient-accounting failure` -> `pass`. This is what lets an engineer tell *which
stage* regressed instead of reading one aggregate score.

### Actual runs (live GPT-4o + the supplied corpus)

Two consecutive runs, same code, same prompt, same 10 queries:

```
Run 1:  Recall@5 1.00  Precision@5 0.24  Hit rate@5 1.00  MRR 1.00
        Selection accuracy 1.00  is_reasonable_match accuracy 1.00  Constraint adherence 1.00
        Matched P/R 0.90/0.90   Missing P/R 0.52/0.57

Run 2:  Recall@5 1.00  Precision@5 0.24  Hit rate@5 1.00  MRR 1.00
        Selection accuracy 0.80  is_reasonable_match accuracy 1.00  Constraint adherence 1.00
        Matched P/R 0.97/0.92   Missing P/R 0.50/0.62
```

Retrieval is essentially perfect and **stable** at this corpus size (expected - 20
well-separated recipes, top_k=5) - identical across both runs. Generation is not: selection
accuracy moved from 1.00 to 0.80 between two runs with nothing changed on our end. The
regressing case in run 2 is a genuinely bad pick, not just an annotation disagreement -
for *"I have chicken breasts and tomatoes, but no cheese please,"* it selected **Stuffed Bell
Peppers** (contains no chicken at all - it uses ground beef - and does contain cheese, the
thing excluded) over the clearly better **Caprese Chicken**, which was sitting right there in
the retrieved candidates both times.

This is the single most important thing the evaluation setup surfaced: **GPT-4o's recipe
selection is not fully deterministic**, so a one-off "it worked when I tried it" run - or even
one saved `results.json` - is not sufficient evidence of quality. It's also *why* `results.json`
is saved as structured data rather than only printed: comparing run 1 against run 2 above only
worked because both were saved, which is exactly the workflow this is meant to support at a
larger scale (comparing before/after a prompt or model change). Missing-ingredient recall
(~0.5-0.6 in both runs) is the other consistent weakness - GPT-4o under-reports a real missing
ingredient (e.g. the parmesan folded into a pesto sauce) more often than it double-counts or
hallucinates one. A production version of this system would need either a lower-temperature/more
constrained generation setup, a larger eval set run multiple times per change (to average out
this variance), or both - not just a single "looks good" pass.

## Image input (bonus, not implemented)

Not implemented - the text pipeline was prioritized per the challenge's guidance, and the
remaining time went into ingestion robustness, the structured-decision guarantees, tests, and
this evaluation fixture instead. If added, per PIPELINE.md section 25, it should be a pure input
adapter, not a second recommendation architecture:

```
image -> GPT-4o vision ingredient extraction (extend custom_components.py's
          ExtractFoodItemsFromImage, preserving uncertainty rather than inventing
          ingredients it can't see)
       -> combine with optional user text (concatenate into one free-text request)
       -> existing RecommendationService.recommend(...) - completely unchanged
```

The `/recommend_recipe` request model would gain an optional image field; the route would call
the vision extraction step first when an image is present, then hand the combined text to the
exact same service used today.

## Future improvements

- Structured ingredient extraction + hybrid ranking (semantic + ingredient-coverage score) once
  the corpus is large enough to need it.
- Grow the human-annotated evaluation set alongside prompt/model changes, and re-run it as a
  regression check.
- LLM-as-judge as a secondary, scalable signal for Markdown quality/usefulness - kept secondary
  to human annotation, per PIPELINE.md.
- A dedicated ingestion job once the corpus outgrows "small enough for startup."
