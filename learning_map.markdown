# RAGAS-FINANCE — Learning Map

## What is this?

A **node-centric RAG system** for SEC financial filings (10-K/10-Q). Ingest EDGAR HTML → section tree → chunk → dense + sparse index → hybrid retrieval → LLM answer.

---

## Architecture Overview

```
┌─ Frontend (Next.js 15) ─────────────────────────────┐
│  API route → proxy to FastAPI backend                │
├─ API & HTTP Layer ──────────────────────────────────┤
│  FastAPI endpoints: ask, finance, health, docs       │
├─ Ask Pipeline ──────────────────────────────────────┤
│  Finance intent → hybrid retrieval → context → LLM  │
├─ Retrieval Backends ────────────────────────────────┤
│  Qdrant (dense)  +  Postgres/OpenSearch (sparse)    │
├─ Finance Domain ────────────────────────────────────┤
│  SQL vs RAG routing, query planning, SEC facts      │
├─ EDGAR Ingestion ───────────────────────────────────┤
│  Parse HTML → section tree → chunk → index          │
├─ Infrastructure ────────────────────────────────────┤
│  Postgres, Qdrant, OpenSearch, Docker               │
└─────────────────────────────────────────────────────┘
```

---

## Guided Tour (12 steps)

### Step 1: Project Overview
Understand the project purpose: ingest SEC EDGAR filings → hybrid retrieval → LLM generation for financial disclosure questions.

### Step 2: Application Entry Point
**File:** `src/agent/run.py`
Thin entry script that inserts `src/agent` on the Python path and delegates to uvicorn.

### Step 3: Central Configuration
**File:** `src/agent/core/config.py`
All runtime settings in one Pydantic class — model selection, embedding provider, retrieval backends, context budgets, finance routing flags, DB connection strings.

### Step 4: FastAPI Server & Routing
**File:** `src/agent/api/server.py`
All HTTP endpoints: health check, ask generation (main Q&A), document catalog, finance query, report persistence, leads extraction.

### Step 5: Finance Intent Routing
**Directory:** `src/agent/tools/finance/`
Classifies questions as needing SQL evidence (quantitative finance, e.g. "What was revenue in 2024?") vs pure RAG (narrative). Rule-first keyword matching with LLM fallback.

### Step 6: Finance SQL Evidence
**Files:**
- `src/agent/tools/finance/financial_facts_repository.py`
- `src/agent/tools/finance/sql_evidence_narrowing.py`
Queries `sec_financial_observations` table for XBRL-tagged facts. Re-ranks RAG hits by matching accession numbers and metric names.

### Step 7: Core RAG Answer Service
**File:** `src/agent/tools/rag_service.py`
Central orchestration: evidence plan → hybrid retrieval → sibling expansion → context budget truncation → optional rerank → LLM generation + quoted evidence.

### Step 8: Hybrid Retrieval & Reranking
**File:** `src/agent/tools/llamaindex_retrieval.py`
Combines dense (Qdrant) + sparse (Postgres/OpenSearch) via Reciprocal Rank Fusion. Optional Bocha reranker and narrative multi-facet rerank.

### Step 9: Retrieval Backends
**Directory:** `src/agent/tools/retrieval_backends/`
Pluggable backends: `factory.py` selects configured adapters. Dense = Qdrant, Sparse = Postgres full-text or OpenSearch.

### Step 10: EDGAR Ingestion Pipeline
**File:** `src/agent/tools/ingestion_service.py`
Parse EDGAR HTML → section tree → enrich (strip stamps, normalize headers) → chunk leaves → write nodes to Postgres → upsert vectors to Qdrant → index sparse text.

### Step 11: Frontend (Next.js 15)
**Directory:** `src/frontend/`
Q&A interface: question input, document scope selector, answer display with evidence quote cards, financial facts tables. Same-origin API route handlers proxy to FastAPI.

### Step 12: Infrastructure & Deployment
**File:** `docker-compose.rag.yml`
Docker containers: Postgres 16, Qdrant, optional OpenSearch, agent + analytics services.

---

## Recommended Learning Path

### Phase 1 — Understand the Config
Start with `src/agent/core/config.py`. Every knob (model, embedding, backends, context budget) is documented there. This makes everything else click faster.

### Phase 2 — Trace One Query End-to-End
Follow the ask pipeline in this order:

1. `src/agent/tools/asks/ask_api.py` — validates params, delegates
2. `src/agent/tools/rag_service.py` — central orchestration
3. `src/agent/tools/finance/finance_intent.py` — SQL vs RAG routing
4. `src/agent/tools/llamaindex_retrieval.py` — hybrid retrieval
5. LLM generation (inside `rag_service.py`)

### Phase 3 — Ingestion Pipeline
- `src/agent/tools/ingestion_service.py` — orchestration
- `src/agent/tools/edgar_htm_parser.py` — EDGAR HTML parsing

### Phase 4 — Frontend
Standard Next.js 15 + Tailwind + Radix UI. Calls backend via API route handlers.

---

## Quick Start Commands

```bash
# Start infra (Postgres + Qdrant)
docker compose -f docker-compose.rag.yml up -d

# Start backend (from src/agent)
venv\Scripts\python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# Start frontend (from src/frontend)
npm install && npm run dev

# Ingest a filing
python scripts/run_sec_finance_pipeline.py ingest-edgar-local \
  --document-id-start 9801 --data-dir tools/data \
  --edgar-glob "EDGAR_320193_*.htm"

# Ask a question (CLI)
python scripts/run_sec_finance_pipeline.py ask-multi \
  --document-ids 9801 --question "What was revenue in 2024?" --top-k 8
```

---

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Node-centric RAG** | Documents are parsed into a section tree; leaf nodes = chunks that get embedded |
| **Hybrid retrieval** | Dense (semantic) + sparse (keyword) search fused via RRF |
| **Finance routing** | SQL for quantitative facts, RAG for narrative descriptions |
| **Section-tree search** | Hit a section node → expand to all its leaf descendants |
| **Sibling expansion** | Include neighboring chunks around a hit for context continuity |
| **Context budget** | `CONTEXT_CHAR_BUDGET` caps total characters fed to the LLM |
| **Evidence quotes** | LLM extracts verbatim quotes from context for UI citation cards |
