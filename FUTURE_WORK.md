# Future Work

## Improve sibling-chunk context expansion

### Current behavior

After reranking, the retrieval pipeline takes the top
`CONTEXT_SIBLING_SEED_COUNT` results as seeds. For each seed,
`fetch_siblings()` queries `level=0` nodes with the same `parent_id`, orders
them by `order_index ASC`, and returns up to `CONTEXT_SIBLING_LIMIT` rows.
The current query is not centered on the seed, so a seed late in a large
section can cause the first N chunks in that section to be fetched rather
than the nearby chunks.

Sibling chunks that were not directly reranked inherit the seed score:

```python
inherited_score = seed_rerank_score * CONTEXT_SIBLING_SCORE_DECAY
```

All siblings inherited from the same seed therefore receive the same score.
This score provides a fallback priority for context selection; it is not an
independent semantic relevance score.

### Problem

When sections are split into many small chunks, the first N same-parent chunks
may be weakly related to the seed or the user question. They can consume
context budget despite not being semantically relevant.

### Candidate solutions

1. **Seed-centered order window**

   Fetch chunks near the seed's `order_index`, for example the four preceding
   and four following chunks. Add `parent_id` and an `order_index` range to
   the SQL query. Decide explicitly whether the seed itself counts toward the
   configured limit.

2. **Seed-centered window followed by reranking**

   First fetch nearby chunks, add them to the candidate pool, and rerun the
   reranker over the expanded pool. This gives better semantic filtering but
   increases reranker latency and cost.

3. **Score refinement for inherited siblings**

   If inherited scores remain in use, consider incorporating distance from the
   seed and/or a lightweight lexical or dense similarity score instead of
   assigning one identical score to every sibling. When the same sibling is
   reached from multiple seeds, merge it using the highest applicable score.

### Decision to make later

Compare the simpler centered window against the more accurate rerank-after-
expansion approach using retrieval quality, answer quality, context size,
latency, and reranker cost. Also verify whether the documented
`CONTEXT_SIBLING_SEED_COUNT=0` behavior should disable expansion; the current
`max(1, ...)` implementation still expands one seed when configured as zero.

## Build a reviewed gold set for RAG evaluation

### Current gap

`tools/data/apple_narrative_questions_100.json` contains questions and topic
tags, but generally does not contain a reference answer or gold evidence. The
current RAGAS run can therefore score faithfulness and reference-free context
precision, but it cannot reliably measure answer correctness or whether the
retriever recalled the evidence that a reviewer considers necessary.

### What the gold set should contain

Each question should be extended with a reviewed evaluation record containing:

1. **Reference answer** — a concise answer grounded only in the SEC filing
   corpus, including the required numbers, periods, causes, caveats, and
   citations. It should be written as an evaluation reference, not as a prompt
   for the production answer.
2. **Gold evidence** — the smallest set of filing passages needed to support
   the reference answer. Record the accession/document identity and preferably
   stable node IDs, plus quoted spans and section/title anchors. If chunk
   boundaries may change, accession + section path + quote/hash should be the
   fallback identity.
3. **Scope constraints** — required filing/form/period (for example, FY2024
   10-K only), accepted equivalent filings, and facts that must not be mixed
   across periods.
4. **Review metadata** — LLM draft, human decision, reviewer notes, and a
   version of the corpus/chunking configuration used for annotation.

The gold evidence is not necessarily one document. A “why did revenue change?”
question may require several passages: a results table, a product-mix
explanation, and a cost or foreign-exchange explanation. However, the evidence
should be minimal and sufficient, rather than labeling an entire 10-K as gold.

### Should gold evidence be retrievable?

Yes, for an end-to-end RAG benchmark the gold evidence must exist in the indexed
corpus and be addressable by the retrieval output. Otherwise a retrieval miss
cannot be distinguished from missing or unindexed source data. The benchmark
should explicitly verify before scoring that every gold passage is present in
the target `document_id`/accession and mapped to one or more current nodes.

Keep two evaluations separate:

```text
Retriever evaluation:
  query → top-k retrieved node IDs
  compare with gold node IDs/spans → recall@k, precision@k, MRR/nDCG

Generator evaluation:
  question + retrieved contexts + answer
  compare with reviewed reference answer/evidence
  → correctness, completeness, citation support, faithfulness
```

This separation is important. If the gold passage is absent from top-k, the
primary failure is retrieval even if the LLM writes a plausible answer. If the
gold passage is retrieved but the answer omits or misstates it, the primary
failure is generation or synthesis. A production answer can still receive
partial credit when it is correct using an accepted equivalent passage.

### Annotation workflow

1. Use an LLM to draft the reference answer, gold quotes, accession, section,
   period, and required facts from the filing corpus.
2. Run validation checks: every quote must occur in the source filing, every
   citation must resolve to an ingested node, and period/form constraints must
   match the question.
3. Have a human reviewer accept, edit, or reject the draft and mark evidence
   as required, supporting, or irrelevant. Do not treat an unreviewed LLM draft
   as ground truth.
4. Store the reviewed records separately from the production question list and
   version them with the corpus/chunking pipeline.
5. Run both retrieval metrics and answer metrics, then compare ordinary versus
   narrative queries and SQL-only versus mixed SQL+RAG questions.

### Proposed record shape

```json
{
  "id": 11,
  "question": "How did FY2024 MD&A explain the change in total net sales?",
  "question_mode": "narrative_only",
  "scope": {"form": "10-K", "period_year": 2024, "accessions": ["..."]},
  "reference_answer": "...",
  "gold_evidence": [
    {
      "accession": "...",
      "document_id": 9801,
      "node_ids": ["..."],
      "section_path": "Item 7 / Net Sales",
      "quote": "...",
      "role": "required"
    }
  ],
  "review": {"status": "approved", "reviewer": "...", "notes": "..."}
}
```

### Evaluation metrics to add

- Retrieval: `recall@k`, `precision@k`, `MRR`/`nDCG`, and section/filing scope
  accuracy against gold evidence.
- Answer: completeness, citation entailment, and `faithfulness`.
- RAGAS metrics to add: `context_recall`, `answer_correctness`, and
  `answer_relevancy`.
- Operations: latency, input/output tokens, reranker calls, context size, and
  cost per question.

### RAGAS metrics currently missing

The current evaluator supports `faithfulness` and reference-free
`context_precision` only. Add the following after the reviewed gold set is
available:

- `context_recall`: whether the retrieved contexts contain the information
  needed to answer the question. Requires a reference answer and/or reviewed
  reference contexts, so it should be interpreted as retrieval coverage rather
  than answer quality alone.
- `answer_correctness`: whether the generated answer agrees with the reviewed
  reference answer. Requires a reference answer and should be supplemented by
  deterministic checks for financial numbers, periods, forms, and filing scope.
- `answer_relevancy`: whether the answer directly addresses the user question
  without irrelevant content. It does not prove factual correctness and should
  be reported separately from `faithfulness` and `answer_correctness`.

Implementation requirements:

1. Extend the reviewed evaluation records with `reference_answer` and reviewed
   reference contexts/evidence.
2. Pass those fields into the RAGAS sample using the installed RAGAS API version;
   do not silently substitute reference-free metrics when the fields are absent.
3. Keep the judge model, prompt/config version, and dataset version fixed when
   comparing embedding or generation models.
4. Report these metrics separately for retrieval and generation, alongside
   human-reviewed samples and operational cost/latency.

RAGAS can provide some answer and faithfulness metrics, but the project should
also retain deterministic checks for citation resolution, required numeric
facts, filing/period scope, and gold-node recall. LLM-as-a-judge scores should
be reported alongside human-reviewed samples rather than treated as the only
quality signal.

## Expose and benchmark retrieval parameters

The previous Milvus implementation exposed BM25 and hybrid-search parameters
such as `bm25_k1`, `bm25_b`, analyzer settings, and RRF `k`. The current
Qdrant/OpenSearch implementation does not expose equivalent controls uniformly:

- OpenSearch uses its default BM25 `k1`/`b`; the project currently configures
  analyzer names and query field boosts such as `title^2.5` and
  `search_hints^4`, which are not substitutes for `k1`/`b`.
- Python implements RRF with a hard-coded smoothing constant `k=60`; this is
  not Milvus `RRFRanker` and is not configurable from the environment.
- Qdrant uses cosine distance and approximate HNSW search with
  `hnsw_ef=128`, `exact=False`; `hnsw_ef` controls dense-search candidate
  breadth, not BM25 or RRF weighting.

Add explicit configuration and offline evaluation for BM25 `k1`/`b`, analyzer
choices, RRF `k`, dense/sparse weights (if weighted fusion is introduced), and
Qdrant HNSW search parameters. Compare Recall@k, context precision, answer
correctness, latency, and cost before changing production defaults.

## RAG Evaluation Delivery Plan

### Step 1 — Offline RAGAS evaluation

Create a reproducible batch benchmark from the reviewed gold set. Extend
`apple_narrative_questions_100.json` with LLM-drafted, human-approved reference
answers and gold filing passages; validate that every gold passage exists in
the indexed corpus before scoring.

The offline benchmark must report retrieval metrics and answer metrics
separately, so it can distinguish corpus coverage problems, retrieval misses,
and generation errors. Include ordinary, narrative, SQL-only, and mixed SQL+RAG
questions, together with latency, token usage, reranker calls, context size,
and cost. Version results with the corpus, chunking configuration, models, and
prompts.

### Step 2 — Online real-time evaluation

Evaluate completed production requests asynchronously after the user response.
Capture the route/evidence plan, retrieved node IDs and contexts, answer,
citations, latency, token usage, and reranker-call count. Apply sampled
RAGAS/LLM-as-a-judge scoring where appropriate, while retaining deterministic
checks for citation resolution, filing/period scope, required numeric facts,
and canary questions with reviewed gold evidence.

The online evaluator must not block the ask path. Add sampling, rate limits,
timeouts, retry budgets, redaction, judge-model versioning, and graceful
fallback when RAGAS or the judge LLM is unavailable. Publish scores and cost /
latency distributions to Langfuse or the evaluation store and compare them
against the offline baseline for regression alerts.

## Low priority — Manage document groups from the frontend

### Goal

Replace the manually maintained `tools/data/document_groups.json` workflow with
a database-backed group management flow. Users should be able to create a new
group in the frontend and select only document IDs that actually exist in the
database after ingestion.

### Proposed design

Add PostgreSQL tables, for example:

```sql
CREATE TABLE rag_document_groups (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE rag_document_group_members (
    group_id UUID NOT NULL REFERENCES rag_document_groups(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (group_id, document_id)
);
```

The frontend should load selectable document IDs from the existing document
catalog endpoint, create/update groups through backend endpoints, and use the
database group membership when calling `ask-multi`. The existing JSON file can
remain as a migration/bootstrap source during the transition, but should not
remain the runtime source of truth.

### Suggested API surface

- `GET /agent/api/document-groups` — list groups and member counts.
- `POST /agent/api/document-groups` — create a named group.
- `GET /agent/api/document-groups/{group_id}` — return group members.
- `PUT /agent/api/document-groups/{group_id}` — replace members with valid document IDs.
- `DELETE /agent/api/document-groups/{group_id}` — delete the group, not documents.
- `GET /agent/api/documents/catalog` — continue supplying the selectable documents.

### Acceptance criteria

- A user can create a named group from the frontend.
- The document picker only offers IDs present in `rag_documents`.
- A missing or deleted document cannot remain as a valid group member.
- Group membership is persisted in PostgreSQL and survives frontend/backend restarts.
- Selecting a group resolves its current database members before retrieval.
- Deleting a group never deletes the underlying document or RAG nodes.
- Existing JSON groups can be imported once with a migration command.
- Backend validation prevents duplicate names, empty groups, and unknown document IDs.
- Tests cover CRUD, document deletion cascade, group selection, and ask routing.

### Migration and safety notes

Use PostgreSQL as the source of truth for group membership. Do not store groups
only in browser state or local JSON. Migrate existing groups such as A–F after
verifying that every referenced ID exists; report missing IDs instead of
silently creating invalid memberships. This feature is intentionally low
priority because the current JSON workflow is sufficient for development and
does not block ingestion or retrieval.
