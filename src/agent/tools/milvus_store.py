"""Milvus vector store wrapper with node ID alignment (dense + BM25 sparse, Qdrant parity).

Mirrors ``tools/vector_store.py`` semantics for the Milvus backends:

- ``id`` is the application node UUID (shared with PostgreSQL ``rag_nodes.id``);
  ``auto_id=False`` so Milvus never generates IDs — same contract as Qdrant points.
- The collection schema is created ONCE with the full field set (text + BM25
  sparse + dense). Step 1A only queries the dense path, but inserts already
  populate ``text`` (Milvus computes the BM25 sparse vector automatically), so
  Step 1B can flip ``SPARSE_BACKEND=milvus`` without rebuilding the collection
  (Milvus schemas are immutable).
- ``retrieval_fields`` (JSON) stores the per-node keyword filter fields with
  every value normalized to a list of strings; ``metadata_filters`` map to
  ``json_contains_any(...)`` expressions, which reproduces the OR semantics of
  the Qdrant ``MatchAny`` payload filter used by ``vector_store.dense_search``.
- Dense hits return the same dict shape as ``vector_store.dense_search``:
  ``node_id`` + ``dense_score`` + payload fields, with ``text_preview``
  (first 1000 chars) and retrieval fields flattened to the top level.

Deliberately NOT stored here: nothing else. Postgres ``rag_nodes`` stays the
source of truth for full node text/metadata; this collection is a retrieval
index (see docs/MULTIMODAL_MILVUS_MIGRATION.md §3).
"""

from __future__ import annotations

import time
from typing import Any

from core.config import config
from loguru import logger
from pymilvus import DataType, Function, FunctionType, MilvusClient

from .rag_stage_log import log_rag
from .retrieval_fields import RETRIEVAL_INDEX_KEYWORD_FIELDS

_client: MilvusClient | None = None

# Milvus VARCHAR max_length is 65535; keep defensive headroom when truncating.
_MAX_TEXT_CHARS = 65000
_MAX_TITLE_CHARS = 1000
_ID_MAX_LENGTH = 64
_INSERT_BATCH = 200

# metadata key recording how many leading chars of `text` are the enrichment
# prefix (repeated title/search_hints) rather than node body text.
_TEXT_PREFIX_META_KEY = "_milvus_text_prefix_chars"


def build_enriched_text(
    title: str | None,
    search_hints: str | None,
    body: str,
    *,
    title_repeats: int,
    hints_repeats: int,
) -> tuple[str, int]:
    """Build the BM25 text: repeated title/search_hints prefix + body.

    Approximates OpenSearch field boosts (title^2.5 / search_hints^4) inside
    Milvus's single BM25 column via term-frequency repetition (sub-linear
    under BM25 k1 saturation). The prefix always sits at the head so the
    defensive 65k truncation only ever clips body text. Returns
    ``(text, prefix_chars)``; ``prefix_chars == 0`` means no enrichment.
    """
    parts: list[str] = []
    clean_title = str(title or "").strip()
    if clean_title and title_repeats > 0:
        parts.extend([clean_title] * int(title_repeats))
    clean_hints = str(search_hints or "").strip()
    if clean_hints and hints_repeats > 0:
        parts.extend([clean_hints] * int(hints_repeats))
    if not parts:
        return str(body or "")[:_MAX_TEXT_CHARS], 0
    prefix = "\n".join(parts)
    enriched = f"{prefix}\n{body or ''}"[:_MAX_TEXT_CHARS]
    return enriched, len(prefix) + 1


def _node_search_hints(node: dict[str, Any]) -> str:
    """search_hints from the node payload, falling back to persisted metadata."""
    direct = str(node.get("search_hints") or "").strip()
    if direct:
        return direct
    meta = node.get("metadata") or {}
    fields = meta.get("_retrieval_fields") or {}
    return str(fields.get("search_hints") or "").strip()


def _strip_text_prefix(text: str, metadata: Any) -> str:
    """Drop the enrichment prefix so callers (reranker/UI) see pure body text."""
    prefix = int((metadata or {}).get(_TEXT_PREFIX_META_KEY) or 0)
    if prefix > 0 and prefix <= len(text):
        return text[prefix:]
    return text


# Scalar fields returned with every dense hit (mirrors the Qdrant payload).
_OUTPUT_FIELDS = [
    "document_id",
    "parent_id",
    "node_type",
    "level",
    "order_index",
    "title",
    "metadata",
    "retrieval_fields",
    "text",
]


def get_client() -> MilvusClient:
    """Process-wide Milvus client singleton (thread-safe enough for our single-loop usage)."""
    global _client
    if _client is None:
        kwargs: dict[str, Any] = {
            "uri": config.milvus_uri,
            "db_name": config.milvus_database,
            "timeout": 30.0,
        }
        if config.milvus_user:
            kwargs["user"] = config.milvus_user
            kwargs["password"] = config.milvus_password or ""
        _client = MilvusClient(**kwargs)
    return _client


def _analyzer_params(name: str) -> dict[str, Any]:
    """Map MILVUS_TEXT_ANALYZER to Milvus analyzer params."""
    normalized = (name or "english").strip().lower()
    if normalized == "jieba":
        return {"tokenizer": "jieba"}
    if normalized in ("", "standard"):
        return {"type": "standard"}
    # english and any other built-in analyzer type passes through as-is.
    return {"type": normalized}


def ensure_collection(vector_size: int | None = None) -> None:
    """Create the collection on first use; no-op afterwards (idempotent)."""
    client = get_client()
    if client.has_collection(config.milvus_collection):
        return
    size = int(vector_size or config.embedding_dimension)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=_ID_MAX_LENGTH)
    schema.add_field("document_id", DataType.INT64)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=_ID_MAX_LENGTH, nullable=True)
    schema.add_field("node_type", DataType.VARCHAR, max_length=_ID_MAX_LENGTH, nullable=True)
    schema.add_field("level", DataType.INT64, nullable=True)
    schema.add_field("order_index", DataType.INT64, nullable=True)
    schema.add_field("title", DataType.VARCHAR, max_length=2000, nullable=True)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params=_analyzer_params(config.milvus_text_analyzer),
    )
    schema.add_field("metadata", DataType.JSON, nullable=True)
    schema.add_field("retrieval_fields", DataType.JSON, nullable=True)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=size)
    # Step 1B path: Milvus derives the BM25 sparse vector from `text` on insert.
    schema.add_function(
        Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    index_params.add_index(field_name="document_id", index_type="INVERTED")
    client.create_collection(
        collection_name=config.milvus_collection,
        schema=schema,
        index_params=index_params,
    )
    logger.info("[MilvusStore] Created collection {} (dim={})", config.milvus_collection, size)


def _as_int_document_id(value: Any) -> int:
    """Coerce document_id to int; bool is rejected explicitly (bool is an int subclass)."""
    if isinstance(value, bool):
        raise TypeError(f"document_id must be an int, got bool: {value!r}")
    return int(value)


def _normalize_filter_values(value: Any) -> list[str]:
    """Normalize a retrieval field value to a non-empty list of strings.

    ``build_retrieval_fields`` produces mixed shapes (scalar strings and lists);
    storing everything as string arrays lets ``json_contains_any`` express the
    same OR-match the Qdrant ``MatchAny`` filter had for both shapes.
    """
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _node_retrieval_fields(node: dict[str, Any]) -> dict[str, list[str]]:
    """Extract keyword retrieval fields from a node dict as normalized string arrays."""
    out: dict[str, list[str]] = {}
    # Keyword fields only: text fields (search_hints/section_path_text) feed the
    # BM25 text enrichment instead; they are not equality-filter targets.
    for field_name in RETRIEVAL_INDEX_KEYWORD_FIELDS:
        values = _normalize_filter_values(node.get(field_name))
        if values:
            out[field_name] = values
    return out


def _node_to_row(node: dict[str, Any]) -> dict[str, Any]:
    """Map a NodeIndexRecord payload to a Milvus row (UUID PK, COSINE dense vector)."""
    parent_id = node.get("parent_id")
    level = node.get("level")
    order_index = node.get("order_index")
    title = str(node.get("title") or "").strip()[:_MAX_TITLE_CHARS] or None
    metadata = node.get("metadata") or {}
    enriched_text, prefix_chars = build_enriched_text(
        title,
        _node_search_hints(node),
        str(node.get("text") or ""),
        title_repeats=config.milvus_text_title_repeats,
        hints_repeats=config.milvus_text_hints_repeats,
    )
    if prefix_chars:
        metadata = {**metadata, _TEXT_PREFIX_META_KEY: prefix_chars}
    return {
        "id": str(node["node_id"]),
        "document_id": _as_int_document_id(node["document_id"]),
        "parent_id": str(parent_id).strip() or None if parent_id is not None else None,
        "node_type": str(node.get("node_type") or "").strip() or None,
        "level": int(level) if level is not None else None,
        "order_index": int(order_index) if order_index is not None else None,
        "title": title,
        "text": enriched_text,
        "metadata": metadata,
        "retrieval_fields": _node_retrieval_fields(node),
        "dense": node["vector"],
    }


def insert_nodes(nodes: list[dict[str, Any]]) -> None:
    """Insert node rows (assumes a preceding delete_document_nodes call)."""
    if not nodes:
        return
    ensure_collection(len(nodes[0]["vector"]))
    client = get_client()
    rows = [_node_to_row(node) for node in nodes]
    for start in range(0, len(rows), _INSERT_BATCH):
        client.insert(
            collection_name=config.milvus_collection,
            data=rows[start : start + _INSERT_BATCH],
        )


def delete_document_nodes(document_id: int) -> None:
    """Delete every Milvus row belonging to one document (filter delete)."""
    client = get_client()
    if not client.has_collection(config.milvus_collection):
        return
    client.delete(
        collection_name=config.milvus_collection,
        filter=f"document_id == {int(document_id)}",
    )


def _quote(value: str) -> str:
    """Quote a string for a Milvus filter expr; escape backslash and double-quote."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_filter_expr(
    document_ids: list[int],
    *,
    levels: list[int] | None = None,
    parent_ids: list[str] | None = None,
    metadata_filters: dict[str, list[str]] | None = None,
) -> str:
    """Build the Milvus boolean expr equivalent of the Qdrant must-conditions.

    document_id/level/parent_id use `in [...]` (MatchAny parity); metadata
    filters use json_contains_any over the normalized string arrays stored in
    `retrieval_fields` (OR semantics over both scalar and list source values).
    """
    doc_ids: list[str] = []
    for raw in document_ids:
        try:
            doc_ids.append(str(_as_int_document_id(raw)))
        except (TypeError, ValueError):
            continue
    if not doc_ids:
        return ""
    clauses = [f"document_id in [{', '.join(doc_ids)}]"]
    clean_levels = sorted({int(v) for v in (levels or [])})
    if clean_levels:
        clauses.append(f"level in [{', '.join(str(v) for v in clean_levels)}]")
    clean_parent_ids = sorted({str(v).strip() for v in (parent_ids or []) if str(v).strip()})
    if clean_parent_ids:
        clauses.append(f"parent_id in [{', '.join(_quote(v) for v in clean_parent_ids)}]")
    for key, values in (metadata_filters or {}).items():
        clean = [str(v).strip() for v in (values or []) if str(v).strip()]
        if clean:
            clauses.append(
                f'json_contains_any(retrieval_fields[{_quote(key)}], '
                f"[{', '.join(_quote(v) for v in clean)}])"
            )
    return " and ".join(clauses)


def dense_search(
    query_vector: list[float],
    *,
    document_ids: list[int],
    limit: int,
    levels: list[int] | None = None,
    parent_ids: list[str] | None = None,
    metadata_filters: dict[str, list[str]] | None = None,
    log_stage: str | None = None,
) -> list[dict[str, Any]]:
    """Dense ANN search with the same contract as vector_store.dense_search."""
    if not query_vector or not document_ids:
        if log_stage:
            log_rag(log_stage, returned=0, reason="no_vector_or_documents", limit=limit, levels=levels)
        return []
    ensure_collection(len(query_vector))
    expr = build_filter_expr(
        document_ids,
        levels=levels,
        parent_ids=parent_ids,
        metadata_filters=metadata_filters,
    )
    if not expr:
        if log_stage:
            log_rag(log_stage, returned=0, reason="no_valid_document_ids", limit=limit, levels=levels)
        return []

    t0 = time.perf_counter()
    response = get_client().search(
        collection_name=config.milvus_collection,
        data=[query_vector],
        anns_field="dense",
        search_params={"metric_type": "COSINE"},
        limit=limit,
        filter=expr,
        output_fields=_OUTPUT_FIELDS,
    )
    hits = response[0] if response else []
    results: list[dict[str, Any]] = []
    for hit in hits:
        entity = dict(hit["entity"] or {})
        text = _strip_text_prefix(str(entity.get("text") or ""), entity.get("metadata"))
        retrieval_fields = entity.get("retrieval_fields") or {}
        results.append(
            {
                "node_id": str(hit["id"]),
                "dense_score": hit["distance"],
                "document_id": entity.get("document_id"),
                "parent_id": entity.get("parent_id"),
                "node_type": entity.get("node_type"),
                "level": entity.get("level"),
                "order_index": entity.get("order_index"),
                "title": entity.get("title"),
                "text_preview": text[:1000],
                "metadata": entity.get("metadata") or {},
                **{str(k): v for k, v in dict(retrieval_fields).items()},
            }
        )
    if log_stage:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        top_scores = [round(float(r["dense_score"]), 6) for r in results[:3]]
        log_rag(
            log_stage,
            returned=len(results),
            limit=limit,
            levels=levels,
            document_ids=len(document_ids),
            parent_ids=len(parent_ids) if parent_ids else None,
            metadata_filter_keys=sorted((metadata_filters or {}).keys()) or None,
            top_dense_scores=top_scores or None,
            latency_ms=elapsed_ms,
        )
    return results


def sparse_search(
    query: str,
    *,
    document_ids: list[int],
    limit: int,
    levels: list[int] | None = None,
    parent_ids: list[str] | None = None,
    metadata_filters: dict[str, list[str]] | None = None,
    query_plan: Any = None,
    log_stage: str | None = None,
) -> list[dict[str, Any]]:
    """BM25 search over the ``text`` field (Step 1B sparse path).

    Milvus analyzes the raw query text server-side with the collection
    analyzer and scores it against the BM25 function's sparse vectors —
    the same rows the dense backend writes. Hit contract matches the
    PostgreSQL/OpenSearch sparse backends: ``node_id`` + ``sparse_score``
    plus payload fields, with full ``text`` (PG stays the source of truth).

    OpenSearch-style per-field boosts from a ``SparseQueryPlan`` cannot map
    onto a single BM25 field, so the plan is logged as not applied
    (docs/MULTIMODAL_MILVUS_MIGRATION.md §3.9).
    """
    normalized = (query or "").strip()
    if not document_ids or not normalized:
        if log_stage:
            log_rag(log_stage, returned=0, reason="no_query_or_documents", limit=limit, levels=levels)
        return []
    ensure_collection()
    expr = build_filter_expr(
        document_ids,
        levels=levels,
        parent_ids=parent_ids,
        metadata_filters=metadata_filters,
    )
    if not expr:
        if log_stage:
            log_rag(log_stage, returned=0, reason="no_valid_document_ids", limit=limit, levels=levels)
        return []

    t0 = time.perf_counter()
    response = get_client().search(
        collection_name=config.milvus_collection,
        data=[normalized],
        anns_field="sparse",
        search_params={"metric_type": "BM25"},
        limit=limit,
        filter=expr,
        output_fields=_OUTPUT_FIELDS,
    )
    hits = response[0] if response else []
    results: list[dict[str, Any]] = []
    for hit in hits:
        entity = dict(hit["entity"] or {})
        text = _strip_text_prefix(str(entity.get("text") or ""), entity.get("metadata"))
        retrieval_fields = entity.get("retrieval_fields") or {}
        results.append(
            {
                "node_id": str(hit["id"]),
                "sparse_score": float(hit["distance"]),
                "document_id": entity.get("document_id"),
                "parent_id": entity.get("parent_id"),
                "node_type": entity.get("node_type"),
                "level": entity.get("level"),
                "order_index": entity.get("order_index"),
                "title": entity.get("title"),
                "text": text,
                "text_preview": text[:1000],
                "metadata": entity.get("metadata") or {},
                **{str(k): v for k, v in dict(retrieval_fields).items()},
            }
        )
    if log_stage:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        top_scores = [round(float(r["sparse_score"]), 6) for r in results[:3]]
        log_rag(
            log_stage,
            returned=len(results),
            limit=limit,
            levels=levels,
            document_ids=len(document_ids),
            parent_ids=len(parent_ids) if parent_ids else None,
            metadata_filter_keys=sorted((metadata_filters or {}).keys()) or None,
            query_len=len(normalized),
            query_profile=getattr(query_plan, "profile", None),
            query_plan_applied=False,
            top_sparse_scores=top_scores or None,
            latency_ms=elapsed_ms,
        )
    return results


def rebuild_texts(
    *,
    document_ids: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rewrite every row's ``text`` under the current enrichment config (upsert).

    Self-contained round-trip: rows (including ``dense``) are read back from
    Milvus, the old enrichment prefix is stripped, the text is re-built from
    the row's own ``title`` + ``metadata._retrieval_fields.search_hints``, and
    the row is upserted — ``dense`` passes through untouched, the BM25 sparse
    vector is regenerated server-side from the new ``text``. Idempotent:
    re-running with unchanged config produces unchanged rows.

    Postgres is never touched; node UUIDs stay stable.
    """
    ensure_collection()
    client = get_client()
    collection = config.milvus_collection
    if document_ids is None:
        probe = client.query(
            collection_name=collection,
            filter="document_id > 0",
            output_fields=["document_id"],
            limit=16384,
        )
        document_ids = sorted({int(row["document_id"]) for row in probe})

    stats: dict[str, Any] = {
        "docs": len(document_ids),
        "rows_seen": 0,
        "rows_changed": 0,
        "rows_upserted": 0,
        "dry_run": dry_run,
        "title_repeats": config.milvus_text_title_repeats,
        "hints_repeats": config.milvus_text_hints_repeats,
        "changed_by_doc": {},
    }
    pending: list[dict[str, Any]] = []

    def _flush() -> None:
        for start in range(0, len(pending), _INSERT_BATCH):
            client.upsert(
                collection_name=collection,
                data=pending[start : start + _INSERT_BATCH],
            )
        stats["rows_upserted"] += len(pending)
        pending.clear()

    for document_id in document_ids:
        rows = client.query(
            collection_name=collection,
            filter=f"document_id == {int(document_id)}",
            output_fields=[*_OUTPUT_FIELDS, "dense"],
            limit=16384,
        )
        changed_this_doc = 0
        for row in rows:
            stats["rows_seen"] += 1
            node_id = row["id"]
            metadata = dict(row.get("metadata") or {})
            raw_text = str(row.get("text") or "")
            old_prefix = int(metadata.get(_TEXT_PREFIX_META_KEY) or 0)
            body = raw_text[old_prefix:] if 0 < old_prefix <= len(raw_text) else raw_text
            title = str(row.get("title") or "").strip()
            hints = str((metadata.get("_retrieval_fields") or {}).get("search_hints") or "").strip()
            enriched, prefix_chars = build_enriched_text(
                title,
                hints,
                body,
                title_repeats=config.milvus_text_title_repeats,
                hints_repeats=config.milvus_text_hints_repeats,
            )
            if enriched == raw_text:
                continue
            stats["rows_changed"] += 1
            changed_this_doc += 1
            new_metadata = {k: v for k, v in metadata.items() if k != _TEXT_PREFIX_META_KEY}
            if prefix_chars:
                new_metadata[_TEXT_PREFIX_META_KEY] = prefix_chars
            pending.append(
                {
                    "id": node_id,
                    "document_id": int(row["document_id"]),
                    "parent_id": row.get("parent_id"),
                    "node_type": row.get("node_type"),
                    "level": row.get("level"),
                    "order_index": row.get("order_index"),
                    "title": row.get("title"),
                    "text": enriched,
                    "metadata": new_metadata,
                    "retrieval_fields": row.get("retrieval_fields") or {},
                    "dense": row["dense"],
                }
            )
            if len(pending) >= _INSERT_BATCH:
                if not dry_run:
                    _flush()
                else:
                    stats["rows_upserted"] += len(pending)
                    pending.clear()
        if changed_this_doc:
            stats["changed_by_doc"][str(document_id)] = changed_this_doc
    if pending:
        if not dry_run:
            _flush()
        else:
            stats["rows_upserted"] += len(pending)
            pending.clear()
    logger.info(
        "[MilvusStore] rebuild_texts done: {} rows seen / {} changed / {} upserted (dry_run={}, title_x{}, hints_x{})",
        stats["rows_seen"],
        stats["rows_changed"],
        stats["rows_upserted"],
        dry_run,
        config.milvus_text_title_repeats,
        config.milvus_text_hints_repeats,
    )
    return stats
