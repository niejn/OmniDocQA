# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project overview

OmniDocQA (formerly RAGAS-FINANCE) is an evaluation-first, node-centric RAG system. It started as SEC-style financial filings QA and grew into a general multimodal document QA platform: text documents are ingested into a section tree (leaf nodes = chunks) in Postgres with dense+BM25 vectors in Milvus; arbitrary PDFs go through a multimodal pipeline (dots.ocr/VLM/fitz → text+image chunks → `rag_multimodal` collection). Answers come from hybrid retrieval (RRF fusion) + local cross-encoder reranking + LLM generation with evidence cards, and the whole stack is measurable via RAGAS metrics with hard-assertion gates. A Next.js frontend provides the UI (`/` for SEC QA, `/documents` for the document library).

## Commands

### Infrastructure (from repo root)

```bash
docker compose -f docker-compose.rag.yml up -d    # Postgres + Milvus (+ rag-minio asset store)
```

### Backend (src/agent)

```bash
# Windows
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# Unix
venv/bin/pip install -r requirements.txt
venv/bin/python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Frontend (src/frontend)

```bash
npm install && npm run dev     # → http://localhost:3000
```

### Ingest (from src/agent)

```bash
# Local EDGAR .htm files
venv\Scripts\python scripts\run_sec_finance_pipeline.py ingest-edgar-local \
  --document-id-start 9801 --data-dir tools\data \
  --edgar-glob "EDGAR_320193_*.htm" --companyfacts-json tools\data\CIK0000320193.json

# Download + ingest from SEC
venv\Scripts\python scripts\run_sec_finance_pipeline.py ingest-edgar \
  --document-id-start 9100 --max-filings 5 --json-path tools\data\CIK0000320193.json

# Reindex vectors only (after embedding model/dimension change)
venv\Scripts\python scripts\run_sec_finance_pipeline.py reindex-vectors --document-id <id>
```

### CLI ask (from src/agent)

```bash
venv\Scripts\python scripts\run_sec_finance_pipeline.py ask-multi \
  --document-ids 9801 --question "..." --top-k 8
```

### Evaluation (requires API running, from src/agent)

```bash
python scripts/run_mixed_narrative_questions_parallel.py --questions tools/data/apple_narrative_questions_100.json
python scripts/run_evaluate_pending_parallel.py
```

### Lint & test

```bash
uv run ruff check tests                    # lint
uv run mypy --strict src/agent/tools/finance/report_locale.py
uv run pytest tests/unit_tests             # unit tests
uv run pytest tests/integration_tests      # integration tests (needs API keys)
```

## Architecture

### Configuration

All settings live in `src/agent/core/config.py` as a Pydantic `Config` class, loaded from `src/agent/.env`. See `src/agent/env.example` for all available vars. The config uses `extra="ignore"` so unknown vars don't cause startup failures.

Key config dimensions:
- **Model selection**: `DEFAULT_MODEL` (e.g. `deepseek/deepseek-chat`, `openai/gpt-4o`). API keys: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `QWEN_API_KEY`, `ANTHROPIC_API_KEY`.
- **Embedding**: `EMBEDDING_PROVIDER` (auto/qwen/openai), `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` (must match Milvus collection).
- **Retrieval backends**: `DENSE_BACKEND=milvus|milvus_multimodal`, `SPARSE_BACKEND=milvus|postgres|none`. Fusion: `FUSION_BACKEND=app` (default) or `milvus` hybrid_search.
- **Context assembly**: `CONTEXT_CHAR_BUDGET` (char-based budget mode), `CONTEXT_SIBLING_*` (sibling expansion), `SECTION_TREE_SEARCH_DEPTH`.
- **Finance routing**: `FINANCE_SQL_ROUTING_ENABLED`, `FINANCE_LLM_ROUTE_ONLY`, `FINANCE_SQL_NARROW_RAG_*`.

### Ask pipeline (the core query path)

1. **Request** → `src/agent/tools/asks/ask_api.py` validates params and delegates to `rag_service.answer_question()`
2. **Finance intent** → `src/agent/tools/finance/finance_intent.py` routes the question: rule-first keyword matching (`question_router.py`), LLM fallback for ambiguous cases. Produces a `FinanceRoute` (need_sql, need_rag) and optionally a `FinanceQueryPlan` via LLM (`finance_query_plan_llm.py`).
3. **Retrieval** → `src/agent/tools/llamaindex_retrieval.py`: hybrid search combining dense (`retrieval_backends/dense_milvus.py`) and sparse (Milvus BM25 via `retrieval_backends/sparse_milvus.py`, or Postgres full-text via `sparse_postgres.py`), fused by RRF (k=60) in-process or pushed down to Milvus `hybrid_search` when `FUSION_BACKEND=milvus`. Narrative queries use section-tree search (hit section nodes → expand to leaf descendants).
4. **Context assembly**: sibling expansion around top seeds, char-budget-based truncation (`CONTEXT_CHAR_BUDGET`), optional title-match guarantees for `narrative_targets`.
5. **Rerank** (optional): local CrossEncoder reranker (`local_reranker.py`), multi-facet rerank for narrative (`narrative_multi_rerank.py`).
6. **SQL evidence** (finance): when `need_sql=true`, queries `sec_financial_observations` table (`financial_facts_repository.py`) and optionally narrows RAG hits by matching accessions/metrics (`sql_evidence_narrowing.py`).
7. **Answer generation** → LLM over assembled context in `rag_service.py`. Optional LLM evidence extraction (`answer_evidence_quotes.py`) for UI quote cards.
8. **Optional**: RAGAS evaluation jobs enqueued to `rag_evaluation_jobs` table, consumed by `run_evaluate_pending_parallel.py`.

### Ingest pipeline

`src/agent/tools/ingestion_service.py` orchestrates: parse EDGAR HTML (`edgar_htm_parser.py` + `edgar_htm_enricher.py`) → build section tree → chunk leaves → write nodes to Postgres (`node_repository.py`) → upsert dense vectors to Milvus → index sparse text (Milvus BM25 or Postgres). Optional companyfacts JSON alignment for metadata (form, filing date, entity name). PDF ingestion for the document library is a separate pipeline: `multimodal_ingest.ingest_one_pdf` (dots.ocr/vLLM parse with fitz fallback → heading/semantic chunking → VLM image description → ark multimodal embeddings → `rag_multimodal` collection + assets in local disk/MinIO), exposed via `scripts/ingest_multimodal_pdf.py` and `POST /agent/api/documents/upload`.

### Key data stores

- **Postgres** (`rag_documents`, `rag_nodes`, `rag_ingest_runs`, `sec_financial_observations`, `rag_evaluation_jobs`, `document_sets`, `document_collections`): node storage, sparse full-text search, SEC facts, eval queue, document-library sets and dynamic collections.
- **Milvus** (`rag_nodes` 1536-dim text+BM25, `rag_multimodal` 2048-dim): dense vector search for both pipelines.
- **MinIO** (optional, `rag-minio` on host 9002/9003): multimodal page/figure assets; default asset store is local disk (`MULTIMODAL_ASSET_STORE=local|minio`).

### Retrieval backends

`src/agent/tools/retrieval_backends/factory.py` selects backends based on config. Backends implement `DenseBackend` / `SparseBackend` protocols (`types.py`). Sparse query profiles (`sparse_query_profiles.py`) define field weights and query construction per domain.

### Finance domain (`src/agent/tools/finance/`)

- `question_router.py` — Rule-first keyword matching for SQL vs RAG routing
- `finance_intent.py` — LLM-driven intent classification + row budget
- `finance_query_plan.py` / `finance_query_plan_llm.py` — Heuristic and LLM-based SQL query planning
- `financial_facts_repository.py` — Query `sec_financial_observations` table
- `sql_evidence_narrowing.py` — Re-rank RAG hits using SQL row metadata
- `sec_company_facts.py` — Parse SEC companyfacts JSON
- `product_surface.py` — Product-level spec (report locale, feature flags)

### Frontend (`src/frontend/`)

Next.js 15 app with Tailwind CSS + Radix UI. Calls backend through a same-origin API route (`/api/ask/generate`) that proxies to the FastAPI backend (`BACKEND_API_BASE_URL` in `.env.local`). The UI shows question input, answer conclusion, evidence cards, financial facts, and filing metadata.

### Project package layout

The Python package root is `src/agent` (configured in `pyproject.toml` via `tool.setuptools.packages.find`). Imports use package-relative paths like `from tools.rag_service import answer_question`. Test config sets `pythonpath = ["src/agent"]` so tests import the same way.
