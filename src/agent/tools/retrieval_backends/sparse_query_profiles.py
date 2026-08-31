"""Structured sparse query profiles for OpenSearch.

Finance-only deployment. To add another domain later:
- Add keyword fields in ``retrieval_fields.RETRIEVAL_INDEX_KEYWORD_FIELDS``.
- Extend ``build_retrieval_fields`` / ``_extract_domain`` for that domain.
- Add a ``_build_<domain>_plan`` and branch in ``build_sparse_query_plan``.
- Route indices in ``sparse_opensearch._index_for_domain`` / ``_configured_index_names``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from core.config import config

_FINANCE_SECTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("management_discussion", ("md&a", "management discussion", "results of operations", "管理层讨论")),
    ("liquidity", ("liquidity", "capital resources", "流动性", "资本资源")),
    ("risk_factors", ("risk factors", "风险因素")),
    ("business_overview", ("business overview", "业务概览", "业务概况")),
)


@dataclass
class SparseQueryPlan:
    """稀疏(关键词/全文)检索查询计划，交给 OpenSearch 执行——不是稀疏向量(sparse-vector)。

    本仓库的 "sparse" 指传统词法 BM25 检索，与 Milvus 的 sparse-vector 内积不同：
    OpenSearch 在 title / text / search_hints 等文本字段上做全文匹配，不存储 token→权重向量。
    字段含义（均来自被拆分的 EDGAR 章节节点）：
      profile : 检索配置名（"finance_v1" 金融专属 / "generic_v1" 通用兜底）
      filters : 硬过滤（必须满足但不参与打分），如 {"term": {"domain": "finance"}}
      must    : 必须满足的查询（参与打分），通常是 multi_match 全文匹配
      should  : 命中则加分（提升相关度），如金融章节 / term_targets 的 match_phrase
      slots   : 给下游的元数据（domain / 章节 / term），不进入查询
    调用 to_bool_query(base_filters) 转成 OpenSearch 的 {"bool": {...}} JSON。
    """
    profile: str
    filters: list[dict[str, Any]] = field(default_factory=list)
    must: list[dict[str, Any]] = field(default_factory=list)
    should: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)

    def to_bool_query(self, base_filters: list[dict[str, Any]]) -> dict[str, Any]:
        query: dict[str, Any] = {
            "must": list(self.must),
            "filter": [*base_filters, *self.filters],
        }
        if self.should:
            query["should"] = list(self.should)
        return {"bool": query}


def _match_phrase(field: str, value: str, boost: float) -> dict[str, Any]:
    return {"match_phrase": {field: {"query": value, "boost": boost}}}


def _term_should(field: str, value: Any, boost: float) -> dict[str, Any]:
    return {"term": {field: {"value": value, "boost": boost}}}


def _detect_finance_sections(query: str) -> list[str]:
    lowered = (query or "").lower()
    matched: list[str] = []
    for value, aliases in _FINANCE_SECTION_RULES:
        if any(alias.lower() in lowered for alias in aliases):
            matched.append(value)
    return matched


def _build_finance_plan(
    query: str,
    *,
    narrative_targets: tuple[str, ...] = (),
    term_targets: tuple[str, ...] = (),
) -> SparseQueryPlan:
    """构造金融专属计划 finance_v1：在通用全文匹配之上叠加领域加分。

    - filters: domain=finance 硬过滤
    - should 加分: 识别到的金融章节(match_phrase on search_hints, boost 6.0)
      + term_targets(match_phrase on search_hints 4.8 / on title 2.4)
    让"叙述性章节 / XBRL 指标"相关的 chunk 排名更靠前。
    """
    normalized = (query or "").strip()
    sections = list(dict.fromkeys([*_detect_finance_sections(normalized), *narrative_targets]))
    should: list[dict[str, Any]] = [
        _term_should("domain", "finance", 7.0),
        _term_should("content_type", "finance_chunk", 2.0),
        _term_should("content_type", "financial_note", 1.4),
        _term_should("content_type", "financial_statement", 1.2),
    ]
    for section in sections:
        should.append(_match_phrase("search_hints", section, 6.0))
    for term in term_targets:
        clean = str(term).strip()
        if not clean:
            continue
        should.append(_match_phrase("search_hints", clean, 4.8))
        should.append(_match_phrase("title", clean, 2.4))
    return SparseQueryPlan(
        profile="finance_v1",
        filters=[{"term": {"domain": "finance"}}],
        must=[
            {
                "multi_match": {
                    "query": normalized,
                    "fields": ["title^2.5", "search_hints^4", "text"],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        ],
        should=should,
        slots={
            "domain": "finance",
            "finance_sections": sections,
            "term_targets": [str(term).strip() for term in term_targets if str(term).strip()],
        },
    )


def build_sparse_query_plan(
    query: str,
    *,
    narrative_targets: tuple[str, ...] = (),
    term_targets: tuple[str, ...] = (),
) -> SparseQueryPlan:
    """构造稀疏检索查询计划，按配置 scope 选择检索配置。

    scope 来自 OPENSEARCH_SPARSE_SEARCH_SCOPE，默认 "finance"。
    - "exam" 已废弃：打 warning 并强制回退 "finance"。
    - "finance"/"all"：走 _build_finance_plan，带 domain 硬过滤 + 金融章节/term 的 should 加分。
    - 其它 scope：generic_v1，仅对 title/search_hints/text 做朴素多字段全文匹配，无金融加分。
    narrative_targets / term_targets 来自 EvidencePlan，用于 finance 分支构造加分项。

    # 两阶段："filter 先选候选集 → BM25(multi_match)在候选集内打分"。
    # document_ids/levels 等 filter 被塞入 bool.filter（不计分），
    # multi_match 的 title^2/search_hints^3/text 只对通过 filter 的文档做 BM25 评分。
    # 这与 Milvus 稀疏向量内积完全不同：此处 sparse=BM25 关键词搜索，非向量内积。
    scope = (config.opensearch_sparse_search_scope or "finance").strip().lower()
    """
    scope = (config.opensearch_sparse_search_scope or "finance").strip().lower()
    if scope == "exam":
        logger.warning(
            "[SparseQuery] OPENSEARCH_SPARSE_SEARCH_SCOPE=exam is removed; using finance profile"
        )
        scope = "finance"

    if scope in ("finance", "all"):
        return _build_finance_plan(
            query,
            narrative_targets=narrative_targets,
            term_targets=term_targets,
        )

    normalized = (query or "").strip()
    return SparseQueryPlan(
        profile="generic_v1",
        must=[
            {
                "multi_match": {
                    "query": normalized,
                    "fields": ["title^2", "search_hints^3", "text"],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        ],
    )
