# 📘 OmniDocQA — Onboarding Guide

> Generated from the project knowledge graph (`.ua/knowledge-graph.json`, analyzed at commit `12587093`). The graph is current with `HEAD`.

## 1. Project Overview

**Name:** `rag-api` (repo: OmniDocQA, formerly RAGAS-FINANCE)

**What it is:** An **evaluation-first, node-centric multimodal RAG platform** (started as SEC-style financial filings QA). Text filings are ingested into a section tree (leaf nodes = chunks) in Postgres with dense+BM25 vectors in Milvus; arbitrary PDFs go through a multimodal pipeline (dots.ocr/VLM/fitz → text+image chunks → `rag_multimodal` collection). Answers come from **hybrid retrieval (RRF) + local reranking + LLM generation** with evidence cards, measured via RAGAS metrics with hard-assertion gates. A Next.js frontend provides the UI (`/` SEC QA, `/documents` document library).

**Languages:** Python, TypeScript, JavaScript, CSS, YAML, TOML, JSON, Markdown, Dockerfile, Shell

**Frameworks/Key stack:**
- Backend: **FastAPI**, Pydantic, LangChain, LangGraph, LlamaIndex, OpenAI
- Observability/Eval: **Langfuse**, **RAGAS**
- Frontend: **Next.js**, React, Tailwind CSS, Radix UI, TypeScript
- Infra: Docker, Docker Compose, GitHub Actions

**Package root:** `src/agent` (imports use package-relative paths, e.g. `from tools.rag_service import answer_question`).

## 2. Architecture Layers

The system is organized into the following layers (from the knowledge graph):

| Layer | Description | Key Files |
|---|---|---|
| **API & HTTP Layer** | FastAPI app, ask router, server entry points, HTTP endpoints | `src/agent/api/server.py`, `tools/asks/ask_api.py`, `src/start_fastapi.py`, `src/agent/run_server.py` |
| **Core Configuration** | Central Pydantic settings + env template for all runtime dimensions | `src/agent/core/config.py`, `src/agent/env.example`, `pyproject.toml` |
| **Ask Pipeline & Answer Generation** | RAG answering service: SQL evidence, retrieval, context assembly, LLM generation, evidence extraction | `tools/rag_service.py`, `tools/rag_graph.py`, `tools/answer_evidence_quotes.py`, `tools/llm.py`, `tools/ragas_llm.py` |
| **Retrieval & Reranking** | Hybrid retrieval orchestration + local CrossEncoder & narrative multi-facet rerankers, embedding/vector helpers | `tools/llamaindex_retrieval.py`, `tools/local_reranker.py`, `tools/narrative_multi_rerank.py`, `tools/vectorizer.py` |
| **Retrieval Backends** | Pluggable dense (Qdrant) / sparse (Postgres full-text / OpenSearch) adapters + factory | `tools/retrieval_backends/factory.py`, `dense_qdrant.py`, `sparse_postgres.py`, `sparse_opensearch.py`, `types.py` |
| **Finance Domain** | SQL vs RAG routing, heuristic/LLM query planning, SEC facts, evidence narrowing, locale | `tools/finance/question_router.py`, `finance_intent.py`, `finance_query_plan*.py`, `financial_facts_repository.py`, `sec_company_facts.py`, `report_locale.py` |
| **EDGAR Ingestion** | End-to-end parse → section tree → chunk → persist → index | `tools/ingestion_service.py`, `edgar_htm_parser.py`, `edgar_htm_enricher.py`, `chunk_segmenter.py`, `node_repository.py` |
| **Observability & Evaluation** | Langfuse tracing, report persistence, RAGAS eval jobs | `tools/langfuse_tracing.py`, `report_store.py`, `evaluation_pipeline.py` |
| **Leads Subsystem** | Bocha search client + part-time graduate leads extraction | `tools/leads/bocha_search.py`, `part_time_graduate_leads.py`, `leads_api.py` |
| **CLI Scripts & Tooling** | SEC finance pipeline, batch eval, ingestion, retrieval analysis | `scripts/run_sec_finance_pipeline.py`, `run_evaluate_pending_parallel.py`, etc. |
| **Frontend Application** | Next.js 15 UI + same-origin API proxy routes | `src/frontend/src/app/page.tsx`, `components/DocumentScope.tsx`, `app/api/ask/generate/route.ts` |
| **Frontend Configuration** | Build/tooling config | `package.json`, `tsconfig.json`, `next.config.mjs`, `tailwind.config.ts` |
| **Infrastructure & Deployment** | Dockerfiles, compose stacks, GitHub Actions CI | `Dockerfile.agent`, `docker-compose.rag.yml`, `docker-compose.yml`, `.github/workflows/*` |
| **Tests** | Unit + integration suites | `tests/unit_tests/test_report_locale.py`, `tests/integration_tests/test_placeholder.py` |
## 3. Key Concepts

- **Node-centric RAG** — Ingestion builds a **section tree** from EDGAR HTML; **leaf nodes are chunks**. Retrieval can hit section nodes and expand to leaf descendants for narrative questions. Nodes live in Postgres (`rag_documents`, `rag_nodes`).
- **Dual indexing** — Dense vectors in **Qdrant** (`rag_nodes` collection) + sparse index in **Postgres full-text or OpenSearch**. Combined via **Reciprocal Rank Fusion (RRF)**.
- **Node storage details** — See [`NODE_STORAGE_AND_RETRIEVAL.md`](NODE_STORAGE_AND_RETRIEVAL.md) for the `level`/`parent_id` tree, ID alignment, embedding inputs, and PostgreSQL/Qdrant responsibilities.
- **Finance routing** — A question is classified as needing **SQL evidence** (quantitative, e.g. "revenue in 2024") vs **RAG** (narrative, e.g. business description), using **rule-first keyword matching with LLM fallback** (`question_router.py` → `finance_intent.py`).
- **SQL evidence narrowing** — When SQL routing is chosen, XBRL-tagged facts are queried from `sec_financial_observations`, and RAG hits are **re-ranked by matching accessions/metrics** (`sql_evidence_narrowing.py`).
- **Hybrid context assembly** — Top seeds expanded with **sibling expansion** and **char-budget truncation** (`CONTEXT_CHAR_BUDGET`), with optional title-match guarantees for `narrative_targets`.
- **Optional reranking** — **local CrossEncoder** (Qwen3-Reranker), plus a **multi-facet narrative rerank** that generates facet sub-queries from the evidence plan.
- **Optional LangGraph planner** — `rag_graph.py` can split a question into sub-queries, retrieve each, and merge results.
- **Filing-resolution** — Since filings come from different report years, finance questions resolve the best-matching filing by form/period/recency (`finance_filing_resolver.py`).
- **Evidence quote cards** — The answer references verbatim quotes extracted from retrieved nodes (`answer_evidence_quotes.py`) for UI evidence cards.
- **RAGAS evaluation** — Ask jobs are enqueued to `rag_evaluation_jobs`, consumed by `run_evaluate_pending_parallel.py`, scored with faithfulness/context-precision, and traced in Langfuse.
- **Central config** — Everything is driven from `core/config.py` (`extra="ignore"`), loaded from `src/agent/.env`; see `env.example` for all variables.

## 4. Guided Tour

Follow the recommended learning path:

1. **Project Overview** — Start with `README.md` and `CLAUDE.md` for purpose.
2. **Application Entry Point** — `src/start_fastapi.py` inserts `src/agent` on the path and delegates to `src/agent/run_server.py`.
3. **Central Configuration** — `src/agent/core/config.py` + `env.example`. Understand model/embedding/backend selection before touching retrieval.
4. **FastAPI Server & Routing** — `src/agent/api/server.py` defines all HTTP endpoints; `tools/asks/ask_api.py` is the ask router.
5. **Finance Intent Routing** — `tools/finance/finance_intent.py` + `question_router.py` + `finance_query_plan.py`.
6. **Finance SQL Evidence** — `tools/finance/financial_facts_repository.py` + `sql_evidence_narrowing.py` + `sec_company_facts.py`.
7. **Core RAG Answer Service** — `tools/rag_service.py` + `rag_graph.py` + `answer_evidence_quotes.py` (the central hub).
8. **Hybrid Retrieval & Reranking** — `tools/llamaindex_retrieval.py` + `local_reranker.py` + `narrative_multi_rerank.py`.
9. **Retrieval Backends** — `tools/retrieval_backends/factory.py` + the Qdrant/Postgres/OpenSearch adapters.
10. **EDGAR Ingestion Pipeline** — `tools/ingestion_service.py` → `edgar_htm_parser.py` → `edgar_htm_enricher.py` → `chunk_segmenter.py` → `node_repository.py`.
11. **Frontend Application** — `src/frontend/src/app/page.tsx`, `DocumentScope.tsx`, API proxy routes.
12. **Infrastructure & Deployment** — `docker-compose.rag.yml`, `Dockerfile.agent`, GitHub Actions CI.

## 5. File Map (by layer)

**API & Entry Points**
- `src/start_fastapi.py` — Server entry script; inserts `src/agent` on the path, launches uvicorn.
- `src/agent/run_server.py` — Configures runtime logging and launches the server.
- `src/agent/api/server.py` — FastAPI app; defines all HTTP endpoints (health, ask, ingestion, finance observations, reports).
- `src/agent/tools/asks/ask_api.py` — Ask router (generate, stream, parse, vector-search, health, finance product spec).

**Configuration**
- `src/agent/core/config.py` — Central Pydantic `Config` for all environment settings.
- `src/agent/env.example` — Reference template of every config variable.
- `pyproject.toml` — Package metadata, deps, package discovery under `src/agent`, tool settings.

**Core RAG Pipeline**
- `tools/rag_service.py` — Central orchestration: SQL evidence + retrieval + context assembly + LLM generation + citations + UI bundle.
- `tools/rag_graph.py` — Optional LangGraph multi-step retrieval planner.
- `tools/rag_stage_log.py` — Structured per-stage logging correlated by request ID.
- `tools/answer_evidence_quotes.py` — Post-answer verbatim quote extraction for evidence cards.
- `tools/llm.py` — Unified `get_llm` factory for OpenAI-compatible/DeepSeek/Qwen ChatOpenAI models.
- `tools/ragas_llm.py` — Strips JSON code fences so RAGAS Pydantic validation succeeds.
- `tools/runtime_logging.py` — loguru console/file logging setup.

**Retrieval & Reranking**
- `tools/llamaindex_retrieval.py` — Core hybrid retrieval (RRF, filing-aware limiting, narrative section-tree expansion).
- `tools/local_reranker.py` — Local sentence-transformers CrossEncoder reranker (Qwen3-Reranker).
- `tools/narrative_multi_rerank.py` — Multi-query rerank for narrative questions.
- `tools/narrative_section_policy.py` — Narrative scoring config (penalties, thresholds).
- `tools/llamaindex_callbacks.py` — LlamaIndex stage timing for observability.
- `tools/vectorizer.py` — Embedding provider selection, batching, retry.
- `tools/vector_store.py` — Qdrant collection lifecycle + dense search.
- `tools/retrieval_fields.py` — Enriches nodes with finance retrieval metadata.

**Retrieval Backends**
- `retrieval_backends/factory.py` — Selects backends from config.
- `retrieval_backends/dense_qdrant.py` — Qdrant dense search.
- `retrieval_backends/sparse_postgres.py` / `sparse_opensearch.py` — Sparse full-text search.
- `retrieval_backends/sparse_query_profiles.py`, `types.py` — Query profiles & protocols.

**Finance Domain**
- `finance/question_router.py` — Rule-first SQL/RAG routing.
- `finance/finance_intent.py` — Question classification + row budget.
- `finance/finance_query_plan.py` / `finance_query_plan_llm.py` — Heuristic & LLM query planning.
- `finance/financial_facts_repository.py` — Queries `sec_financial_observations`.
- `finance/sql_evidence_narrowing.py` — Re-ranks RAG hits by SQL row metadata.
- `finance/sec_company_facts.py` — Parses SEC companyfacts JSON into fact rows.
- `finance/finance_filing_resolver.py` — Resolves best-matching filing by form/period/recency.
- `finance/companyfacts_accession_period.py` — Resolves fiscal reporting periods from DEI facts.
- `finance/report_locale.py` — zh/en locale strings (tested).
- `finance/product_surface.py` — Product-facing QA surface & evidence UI bundle.
- `finance/edgar_client.py` / `edgar_sync.py` — SEC EDGAR download & ingest orchestration.

**EDGAR Ingestion**
- `tools/ingestion_service.py` — Full pipeline orchestrator.
- `tools/edgar_htm_parser.py` — DOM parser classifying financial vs narrative elements.
- `tools/edgar_htm_enricher.py` — Element enrichment (strip page stamps, normalize headers).
- `tools/edgar_htm_to_final.py` — One-shot HTM→final JSON pipeline driver.
- `tools/chunk_segmenter.py` — Heading-aware, sentence-aware chunk splitting.
- `tools/node_repository.py` — Postgres repo for docs/nodes/section-tree/sparse-search/eval-queue.
- `tools/document_parser.py` — PDF/TXT/HTML/DOCX parsing dispatch.
- `tools/document_display.py` / `document_groups.py` — Catalog rows & doc groups.

**Observability & Evaluation**
- `tools/langfuse_tracing.py` — Langfuse tracer wrapper.
- `tools/report_store.py` — File-based report persistence.
- `tools/evaluation_pipeline.py` — Runs pending RAGAS jobs (faithfulness, context-precision).

**Frontend**
- `src/frontend/src/app/page.tsx` — Main Q&A UI (question input, evidence cards, financial facts, i18n).
- `src/frontend/src/components/DocumentScope.tsx` — Document scope selector (group or id list).
- `src/frontend/src/app/api/ask/generate/route.ts` — Same-origin proxy to backend.
- Other API routes: `documents/catalog|groups|ids`, `report/[traceId]|delete|list|save`.

**CLI / Scripts**
- `scripts/run_sec_finance_pipeline.py` — Primary SEC finance CLI orchestrator (ingest, ask, reindex, observations).
- `scripts/run_evaluate_pending_parallel.py` — Drains the RAGAS eval queue.
- `scripts/run_mixed_narrative_questions_parallel.py` — Batch narrative eval.
- `scripts/inspect_opensearch.py`, `analyze_retrieval_from_json.py`, `plot_retrieval_compare.py`, etc.

**Infrastructure**
- `docker-compose.rag.yml` — Local Postgres 16 + Qdrant + OpenSearch stack (**the dev dependency**).
- `docker-compose.yml` — Full app stack (Postgres, Redis, agent/analytics, Celery, nginx).
- `Dockerfile.agent` / `Dockerfile.analytics` — Service images.
- `.github/workflows/unit-tests.yml`, `integration-tests.yml` — CI.

## 6. Complexity Hotspots ⚠️

Approach these carefully — they carry the highest complexity:

- **`rag_service.py`** *(complex)* — The heart of the pipeline; touches SQL evidence, retrieval, context assembly, generation, citations, and serialization. Most behavioral coupling lives here.
- **`llamaindex_retrieval.py`** *(complex)* — Hybrid retrieval with RRF, filing-aware limiting, narrative selection, multi-query rerank, section-tree expansion.
- **`node_repository.py` & `ingestion_service.py`** *(complex)* — Ingest/persistence: schema, section tree, sparse search, eval queue.
- **`edgar_htm_parser.py`** *(complex)* — DOM-walking parser with financial vs narrative table classification; subtle heuristics.
- **`chunk_segmenter.py`** *(complex)* — Heading-aware, sentence-aware segmentation with overlap/tail merging.
- **`core/config.py`** *(complex)* — Everything is config-drivable; changing the interface ripples across the whole stack.
- **`vectorizer.py`** *(complex)* — Provider-specific embedding limits/batching/retry.
- **`retrieval_fields.py`** *(complex)* — Heuristic finance metadata enrichment.
- **`financial_facts_repository.py` / `finance_query_plan.py` / `finance_filing_resolver.py` / `companyfacts_accession_period.py`** *(complex)* — Finance-domain heuristics and filing resolution.
- **`langfuse_tracing.py` / `report_store.py`** *(complex)* — Observability plumbing with fallback paths.
- **`page.tsx`** *(complex)* — Rich frontend state: report history, multi-select deletion, i18n following answer locale.
- **`bocha_search.py` / `part_time_graduate_leads.py`** *(complex)* — Leads extraction with nesting normalization.
