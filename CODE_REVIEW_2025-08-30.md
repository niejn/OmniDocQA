# Code Review: RAGAS-FINANCE Ask Pipeline Deep Dive

**Date**: 2025-08-30  
**Scope**: Ask pipeline (intent routing → SQL → hybrid retrieval → rerank → LLM generation)  
**Focus**: Documentation via Chinese comments + architecture clarification

---

## Summary

This review documents the **core ask pipeline** of RAGAS-FINANCE through targeted Chinese comments added to 6 files. The pipeline resolves: question → finance intent routing → structured `FinanceQueryPlan` → parameterized SQL (exact facts) + hybrid BM25/dense retrieval (narrative) → optional rerank → LLM generation.

Key architectural clarifications captured:
1. **SQL path is NOT text-to-SQL** — it's text → structured `FinanceQueryPlan` → parameterized SQL template with bind params
2. **Routing is two independent switches** (`need_sql`/`need_rag`), not mutually exclusive — both can be true (complementary)
3. **"Sparse" = BM25 keyword search** (OpenSearch `multi_match` / Postgres `to_tsvector`), NOT Milvus-style sparse vectors
4. **`document_ids` is a metadata pre-filter** (scope), not query content — symmetric across dense (Qdrant `query_filter`) and sparse (OpenSearch `bool.filter`)
5. **Images/charts are NOT extracted** — only HTML `<TABLE>` financial tables + narrative text; raster images silently dropped
6. **Graceful degradation everywhere** — missing metric keys → broader SQL (no crash); ILIKE fuzzy fallback; LLM fallback on routing ambiguity

---

## Files Modified (Chinese Comments Added)

### 1. `src/agent/tools/vector_store.py` (L152-155)
**Context**: `dense_search()` builds Qdrant `query_filter`
```python
# document_ids 是【元数据预过滤】（payload filter），非向量查询条件。
# 它在 Qdrant 侧通过 query_filter=Filter(must=must_conditions) 先缩小候选集，
# 再在该子集内做向量相似度搜索。与 sparse_opensearch 侧的 terms filter 对称。
```

### 2. `src/agent/tools/retrieval_backends/sparse_opensearch.py` (L243-245)
**Context**: `filters` list construction for OpenSearch
```python
# document_ids 通过 terms filter 先做【候选集裁剪】，随后在候选集内用 BM25（multi_match + ^N boost）打分。
# 这是"过滤→粗排"两阶段：filter 不参与打分，只决定谁有资格被 BM25 评分。
```

### 3. `src/agent/tools/retrieval_backends/sparse_query_profiles.py` (L139-142)
**Context**: `build_sparse_query_plan()` scope dispatch
```python
# 两阶段："filter 先选候选集 → BM25(multi_match)在候选集内打分"。
# document_ids/levels 等 filter 被塞入 bool.filter（不计分），
# multi_match 的 title^2/search_hints^3/text 只对通过 filter 的文档做 BM25 评分。
# 这与 Milvus 稀疏向量内积完全不同：此处 sparse=BM25 关键词搜索，非向量内积。
```

### 4. `src/agent/tools/edgar_htm_parser.py` (L7-8)
**Context**: Module docstring — parsing strategy
```python
图片/图表（<img>、PNG/JPG/SVG）不被提取、不做 OCR/VLM 识别，需另接管线。
HTML <TABLE> 财务数据表会被识别并转为 Markdown（见 _table_to_markdown / _classify_table）。
```

### 5. `src/agent/tools/finance/financial_facts_repository.py` (L360-364)
**Context**: `query_observations_by_filters()` — core SQL executor
```python
    关键：这是【参数化 SQL】，不是 text-to-SQL。
    - SQL 模板固定（_OBS_SELECT + 动态 WHERE），问题文本从不入 SQL 字符串。
    - FinanceQueryPlan 的字段（metric_keys/forms/period_years/accns）经清洗后，
      作为类型化 bind 参数 $N::type[] 填入模板（见 L366-389）。
    - 值均做校验：str.strip、年份≥1900、长度裁剪，再由 pool.fetch(sql, *args) 安全执行。
```

### 6. `src/agent/tools/finance/question_router.py` (L99-107)
**Context**: `route_finance_by_rules()` — routing logic
```python
    规则优先：同时数两类关键词命中数。
    - _SQL_HINTS（如"多少/营收/revenue/净利润"）→ s>0 暗示需结构化数字
    - _RAG_HINTS（如"为什么/如何/解释"）→ r>0 暗示需叙述上下文
    四分支：
      s>0 & r>0 → FinanceRoute(True, True)  # SQL+RAG 互补，最常见
      s>0 & r=0 → FinanceRoute(True, False) # 纯取数
      s=0 & r>0 → FinanceRoute(False, True) # 纯叙述
      s=0 & r=0 → None（歧义→LLM 兜底 resolve_finance_intent/route_finance_with_llm）
    need_sql/need_rag 是独立开关，非互斥；两者同开时管线并行跑 SQL 取数 + RAG 取文，最后融合。
```

---

## Architecture Clarifications (from Discussion)

### The Ask Pipeline (End-to-End)

```
Question
    │
    ├─► _finance_sql_bundle()  ──► route_finance_by_rules() ──► need_sql? need_rag?
    │       │                          │
    │       │                          ├─ s>0,r>0 → both (complementary)
    │       │                          ├─ s>0,r=0 → SQL only
    │       │                          ├─ s=0,r>0 → RAG only
    │       │                          └─ None   → LLM fallback (route_finance_with_llm)
    │       │
    │       └─► build_finance_evidence_plan() ──► FinanceQueryPlan
    │              (heuristic: regex + _PHRASE_TO_TAGS dict + CamelCase fallback)
    │              [optional] LLM override if finance_llm_sql_planner_enabled
    │
    ├─► SQL Path (need_sql)
    │     query_observations_by_filters(document_ids, metric_keys, forms, years, accns, limit)
    │     → exact = ANY($N) filters → parameterized SQL
    │     → fallback: query_observations_by_metric_hints() → ILIKE %hint% (fuzzy, ≥6 chars)
    │     → format_sql_observations_for_prompt() → exact facts to LLM
    │
    ├─► RAG Path (need_rag)
    │     Hybrid retrieval:
    │       Dense: Qdrant vector search (query_filter=document_ids)
    │       Sparse: OpenSearch BM25 (bool.filter=document_ids + multi_match^N)
    │       → RRF fusion → sibling expansion → char-budget truncation
    │
    └─► Fusion & Generation
          prioritize_nodes_by_sql_evidence()  # SQL rows rerank RAG candidates
          LLM generation over sql_context_text + retrieved nodes
```

### SQL vs RAG — Complementary, Not Redundant

| Dimension | SQL (`sec_financial_observations`) | RAG (EDGAR HTML chunks) |
|-----------|-----------------------------------|--------------------------|
| **Source** | SEC companyfacts XBRL JSON (audited facts) | EDGAR HTML (narrative + flattened tables) |
| **Query** | Parameterized `WHERE metric_key='X' AND year=Y` | BM25 + dense vector similarity |
| **Returns** | Exact typed value (`value_numeric=383285000000`) | Text passages mentioning the concept |
| **Precision** | 100% (authoritative) | Fuzzy (recall-oriented) |
| **Use Case** | "Apple 2023 revenue = ?" | "Why did revenue grow?" |

### Parameterized SQL — Not Text-to-SQL

| Text-to-SQL (Classic) | This System |
|----------------------|-------------|
| LLM generates SQL string | LLM/rules produce **fixed-schema JSON** (`FinanceQueryPlan`) |
| Arbitrary SELECT/JOIN/WHERE | **Fixed template** (`_OBS_SELECT` + `WHERE ... = ANY($N)`) |
| Model sees SQL syntax | Model **never sees SQL**; only fills typed bind params |
| Injection/schema drift risk | Impossible — values validated, template fixed |

### `document_ids` — Scope Filter, Not Query Content

- **Dense (Qdrant)**: `query_filter=Filter(must=[FieldCondition(key="document_id", match=MatchAny(...))])`
- **Sparse (OpenSearch)**: `bool.filter=[{"terms": {"document_id": [...]}}]`
- **SQL**: `WHERE document_id = ANY($1::bigint[])`
- All three: **mandatory**, always present, scopes to user-selected filings

### Sparse = BM25, Not Sparse Vectors

- OpenSearch: `multi_match` with `title^2`, `search_hints^3`, `text`
- Postgres: `to_tsvector` + `websearch_to_tsquery`
- **No Milvus**, no SPLADE, no BGE-M3, no inner-product sparse vectors
- `^N` = BM25 field boost (multiplier), not occurrence count
- `search_hints` = synthetic "label bag" (filename + section + role + finance tags + title)

### Graceful Degradation (No Hard Failures)

| Failure Point | Behavior |
|---------------|----------|
| Metric phrase not in `_PHRASE_TO_TAGS` | `metric_keys=[]` → SQL runs without metric filter → broader results |
| Typo/abbrev too short for ILIKE (e.g., "rev") | Falls to broader SQL (no metric filter) |
| Exact SQL returns too few rows | ILIKE fuzzy fallback (`query_observations_by_metric_hints`) |
| Routing rules ambiguous (no hints) | LLM fallback (`finance_sql_routing_llm_fallback=True`) |
| LLM planner fails/returns empty | Falls back to heuristic `build_finance_evidence_plan` |
| LLM router fails | Default hybrid route (`FinanceRoute(True, True)`) |

### Images/Charts — Not Supported

- Parser only handles: **HTML `<TABLE>`** (financial → Markdown; layout → narrative) + **leaf `<div>`** text
- **No `<img>` extraction**, no OCR, no VLM, no chart/figure recognition
- SEC filings use HTML tables for financial data (so most "charts" are actually tables and handled)
- Raster image charts (PNG/JPG trends) would be silently dropped

---

## Outstanding / Follow-up Items

1. **`_PHRASE_TO_TAGS` coverage audit** — Chinese phrases (营收/净利润/毛利率) may be missing; limits heuristic precision
2. **LLM planner eval** — enable `finance_llm_sql_planner_enabled` and measure metric extraction recall vs heuristic
3. **Postgres sparse backend comment** — add equivalent BM25 comment in `node_repository.py` (L87-89, 668-703)
4. **Image pipeline** — if chart extraction needed, design: `<img>` extract → OCR/VLM → text chunk ingest
5. **Routing hint lexicon** — review `_SQL_HINTS`/`_RAG_HINTS` for domain completeness

---

## Verification

All modified files pass `uv run python -m py_compile` (no syntax errors).