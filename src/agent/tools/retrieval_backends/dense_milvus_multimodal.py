"""Milvus multimodal dense backend (T1.4) — new ``rag_multimodal`` collection.

Independent of ``tools/milvus_store.py``/``dense_milvus.py`` by design (hard
constraint: the text chain is untouched). Differences from the text backend:

- Vectors are generated INSIDE the backend at write time (offline ingest) via
  the injected multimodal vectorizer — there are no PG ``rag_nodes`` rows on
  this path (§4.1).
- PK is the chunk UUID (not shared with any PG table).
- Scalar fields ``book_id``/``chapter_label`` are denormalized display fields
  written on every point so evidence cards never join back to PG (§6.1).
- ``text`` carries the BM25 function (sparse is reserved for P2; v1 queries
  dense only). Analyzer is jieba (Chinese courseware; reference collection
  parity) — deliberately independent of the text collection's env knob.
- Search filter is a single Milvus expression: the service layer normalizes
  books+chapters into a document-id UNION, so filtering is
  ``document_id in [...] and kind in [...]`` or ``id in [...]`` (enumerated
  sets) — zero application-side joins (§6.1 retrieval path).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.config import config
from loguru import logger
from pymilvus import DataType, Function, FunctionType

from ..milvus_store import MILVUS_MAX_QUERY_WINDOW, get_client
from ..multimodal_vectorizer import EmbeddingQuotaExceededError, MultimodalVectorizer
from ..rag_stage_log import log_rag

_ID_MAX_LENGTH = 64
_MAX_TEXT_CHARS = 65000  # Milvus VARCHAR hard cap is 65535
_TITLE_MAX_CHARS = 1000
_FILENAME_MAX_CHARS = 500
_LABEL_MAX_CHARS = 1000
_INSERT_BATCH = 100

_OUTPUT_FIELDS = [
    "document_id",
    "filename",
    "title",
    "kind",
    "page_no",
    "text",
    "category",
    "image_ref",
    "book_id",
    "chapter_label",
]


@dataclass
class MmChunkRow:
    """A chunk ready for upsert (text finalized, image asset resolved)."""

    kind: str  # "text" | "image"
    page_no: int
    title: str
    text: str  # body text; image chunks carry the description
    category: str = "Text"
    image_ref: str | None = None  # asset key "{document_id}/{name}"
    image_data_uri: str | None = None  # data URI for embedding (transient, not stored)
    truncated: bool = False
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def _quote(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_filter_expr(
    *,
    document_ids: list[int] | None = None,
    kinds: list[str] | None = None,
    chunk_ids: list[str] | None = None,
) -> str:
    """Single-expression filter: enumerated sets by PK, otherwise doc union + kind."""
    clauses: list[str] = []
    if chunk_ids:
        clean = sorted({str(c).strip() for c in chunk_ids if str(c).strip()})
        if clean:
            clauses.append(f"id in [{', '.join(_quote(c) for c in clean)}]")
    doc_clauses: list[str] = []
    if document_ids:
        ids = sorted({int(d) for d in document_ids})
        if ids:
            doc_clauses.append(f"document_id in [{', '.join(str(i) for i in ids)}]")
    clean_kinds = sorted({str(k).strip() for k in (kinds or []) if str(k).strip()})
    if clean_kinds:
        doc_clauses.append(f"kind in [{', '.join(_quote(k) for k in clean_kinds)}]")
    if doc_clauses:
        clauses.append(" and ".join(doc_clauses))
    return " and ".join(clauses)


class MilvusMultimodalDenseBackend:
    """DenseBackend-protocol-compatible multimodal backend (used directly by the
    /documents service; factory returns it under DENSE_BACKEND=milvus_multimodal)."""

    def __init__(self, collection: str | None = None) -> None:
        self.collection = collection or config.multimodal_collection

    # ── schema ────────────────────────────────────────────────────────
    def ensure_collection(self, vector_size: int | None = None) -> None:
        """Create once with the full schema; validate dimension on every call."""
        client = get_client()
        size = int(vector_size or (config.multimodal_embedding_dim or 0))
        if client.has_collection(self.collection):
            existing = self._collection_dim(client)
            if size and existing and existing != size:
                raise ValueError(
                    f"Collection {self.collection!r} dim={existing} != embedding dim={size}; "
                    "switching multimodal embedding models requires dropping/rebuilding the collection"
                )
            return
        if not size:
            raise ValueError("multimodal collection does not exist and vector dim is unknown yet")
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=_ID_MAX_LENGTH)
        schema.add_field("document_id", DataType.INT64)
        schema.add_field("filename", DataType.VARCHAR, max_length=_FILENAME_MAX_CHARS, nullable=True)
        schema.add_field("title", DataType.VARCHAR, max_length=_TITLE_MAX_CHARS, nullable=True)
        schema.add_field("kind", DataType.VARCHAR, max_length=16)
        schema.add_field("page_no", DataType.INT64, nullable=True)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
        )
        schema.add_field("category", DataType.VARCHAR, max_length=100, nullable=True)
        schema.add_field("image_ref", DataType.VARCHAR, max_length=_FILENAME_MAX_CHARS, nullable=True)
        schema.add_field("book_id", DataType.VARCHAR, max_length=_LABEL_MAX_CHARS, nullable=True)
        schema.add_field("chapter_label", DataType.VARCHAR, max_length=_LABEL_MAX_CHARS, nullable=True)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=size)
        # BM25 sparse is generated but unused by v1 queries (§6 reserved field).
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
        index_params.add_index(field_name="book_id", index_type="INVERTED")
        client.create_collection(collection_name=self.collection, schema=schema, index_params=index_params)
        logger.info("[MmBackend] created collection {} (dim={})", self.collection, size)

    def _collection_dim(self, client) -> int | None:
        try:
            desc = client.describe_collection(collection_name=self.collection)
        except Exception:
            return None
        for f in desc.get("fields", []):
            if f.get("name") == "dense":
                return int((f.get("params") or {}).get("dim") or 0)
        return None

    # ── write path ────────────────────────────────────────────────────
    async def upsert_document_nodes(
        self,
        document_id: int,
        chunks: list[MmChunkRow],
        *,
        vectorizer: MultimodalVectorizer,
        filename: str,
        book_id: str,
        chapter_label: str,
    ) -> dict[str, Any]:
        """Vectorize-validate-then-replace (idempotent re-run).

        All chunks are embedded and dimension-checked BEFORE the old
        document's points are deleted: a failed embed or dim mismatch must
        leave the previous points intact instead of wiping them (the old
        delete-first order emptied the document whenever vectorization
        failed mid-way).
        """
        if not chunks:
            return {"inserted": 0, "truncated": 0}
        # 1. Vectorize everything first — fail-fast before ANY destructive write.
        rows: list[dict[str, Any]] = []
        truncated_count = 0
        for chunk in chunks:
            if chunk.kind == "image":
                if not chunk.image_data_uri:
                    raise ValueError(f"image chunk {chunk.chunk_id} missing image_data_uri")
                result = await vectorizer.embed_image_with_text(chunk.image_data_uri, chunk.text)
            else:
                result = await vectorizer.embed_text(f"{chunk.title}：{chunk.text}" if chunk.title else chunk.text)
            if result.vector is None:
                if result.quota_exhausted:
                    # Provider quota is gone until its reset — abort the whole
                    # document immediately (upload API maps this to 503 with the
                    # reset time carried in the message).
                    raise EmbeddingQuotaExceededError(
                        f"embedding aborted at chunk {chunk.chunk_id}: {result.error}"
                    )
                raise ValueError(f"embedding failed for chunk {chunk.chunk_id}: {result.error}")
            if result.truncated:
                truncated_count += 1
            rows.append(
                {
                    "id": chunk.chunk_id,
                    "document_id": int(document_id),
                    "filename": str(filename or "")[:_FILENAME_MAX_CHARS] or None,
                    "title": str(chunk.title or "")[:_TITLE_MAX_CHARS] or None,
                    "kind": chunk.kind,
                    "page_no": int(chunk.page_no),
                    "text": str(chunk.text or "")[:_MAX_TEXT_CHARS],
                    "category": str(chunk.category or "")[:100] or None,
                    "image_ref": chunk.image_ref,
                    "book_id": str(book_id or "")[:_LABEL_MAX_CHARS] or None,
                    "chapter_label": str(chapter_label or "")[:_LABEL_MAX_CHARS] or None,
                    "dense": result.vector,
                }
            )
        # 2. Dimension consistency + collection dim check, still write-free.
        first_dim = len(rows[0]["dense"])
        mismatched = [r["id"] for r in rows if len(r["dense"]) != first_dim]
        if mismatched:
            raise ValueError(f"inconsistent embedding dims in batch: {len(mismatched)} rows off {first_dim}")
        self.ensure_collection(first_dim)
        # 3. Replace: delete the old points only now that the new rows are
        #    fully validated, then insert.
        self.replace_document_nodes(int(document_id))
        client = get_client()
        for start in range(0, len(rows), _INSERT_BATCH):
            client.insert(collection_name=self.collection, data=rows[start : start + _INSERT_BATCH])
        log_rag(
            "mm_upsert",
            document_id=int(document_id),
            inserted=len(rows),
            truncated=truncated_count,
            dim=first_dim,
            collection=self.collection,
        )
        return {"inserted": len(rows), "truncated": truncated_count, "dim": first_dim}

    def replace_document_nodes(self, document_id: int, nodes: list[Any] | None = None) -> None:
        """Delete every point of one document (DenseBackend-protocol signature)."""
        client = get_client()
        if not client.has_collection(self.collection):
            return
        client.delete(collection_name=self.collection, filter=f"document_id == {int(document_id)}")

    # ── read path ─────────────────────────────────────────────────────
    def count(self) -> int:
        client = get_client()
        if not client.has_collection(self.collection):
            return 0
        stats = client.get_collection_stats(collection_name=self.collection)
        try:
            return int((stats or {}).get("row_count") or 0)
        except (TypeError, ValueError):
            return 0

    def has_collection(self) -> bool:
        return get_client().has_collection(self.collection)

    def search(
        self,
        query_vector: list[float],
        *,
        document_ids: list[int] | None = None,
        limit: int = 8,
        kinds: list[str] | None = None,
        chunk_ids: list[str] | None = None,
        log_stage: str | None = None,
        **_ignored: Any,
    ) -> list[dict[str, Any]]:
        """Dense top-k over rag_multimodal; None/empty document_ids = whole library."""
        if not query_vector:
            return []
        self.ensure_collection(len(query_vector))
        expr = build_filter_expr(document_ids=document_ids, kinds=kinds, chunk_ids=chunk_ids)
        t0 = time.perf_counter()
        response = get_client().search(
            collection_name=self.collection,
            data=[query_vector],
            anns_field="dense",
            search_params={"metric_type": "COSINE"},
            limit=int(limit),
            filter=expr or None,
            output_fields=_OUTPUT_FIELDS,
        )
        hits = response[0] if response else []
        results: list[dict[str, Any]] = []
        for hit in hits:
            entity = dict(hit["entity"] or {})
            text = str(entity.get("text") or "")
            results.append(
                {
                    "chunk_id": str(hit["id"]),
                    "score": float(hit["distance"]),
                    "kind": entity.get("kind"),
                    "document_id": entity.get("document_id"),
                    "filename": entity.get("filename"),
                    "title": entity.get("title"),
                    "page_no": entity.get("page_no"),
                    "text_preview": text[:2000],
                    "text_full_chars": len(text),
                    "category": entity.get("category"),
                    "image_ref": entity.get("image_ref") or None,
                    "book_id": entity.get("book_id"),
                    "chapter_label": entity.get("chapter_label"),
                }
            )
        if log_stage:
            log_rag(
                log_stage,
                returned=len(results),
                limit=limit,
                filter_expr=bool(expr),
                document_ids=len(document_ids or []),
                kinds=kinds,
                chunk_ids=len(chunk_ids or []) or None,
                top_scores=[round(r["score"], 4) for r in results[:3]] or None,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        return results

    def query_chunks(
        self,
        *,
        document_id: int,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """MM-4 chapter browsing: page through one document's chunks (no vectors)."""
        client = get_client()
        if not client.has_collection(self.collection):
            return []
        clauses = [f"document_id == {int(document_id)}"]
        if kind:
            clauses.append(f"kind == {_quote(kind)}")
        rows = client.query(
            collection_name=self.collection,
            filter=" and ".join(clauses),
            output_fields=_OUTPUT_FIELDS,
            limit=int(limit),
            offset=int(offset),
        )
        out = []
        for row in rows:
            text = str(row.get("text") or "")
            out.append(
                {
                    "chunk_id": str(row["id"]),
                    "kind": row.get("kind"),
                    "document_id": row.get("document_id"),
                    "filename": row.get("filename"),
                    "title": row.get("title"),
                    "page_no": row.get("page_no"),
                    "text_preview": text[:2000],
                    "category": row.get("category"),
                    "image_ref": row.get("image_ref") or None,
                    "book_id": row.get("book_id"),
                    "chapter_label": row.get("chapter_label"),
                }
            )
        out.sort(key=lambda r: (r["page_no"] or 0, r["chunk_id"]))
        return out

    def count_document_chunks(self, document_id: int, kind: str | None = None) -> int:
        client = get_client()
        if not client.has_collection(self.collection):
            return 0
        clauses = [f"document_id == {int(document_id)}"]
        if kind:
            clauses.append(f"kind == {_quote(kind)}")
        rows = client.query(
            collection_name=self.collection,
            filter=" and ".join(clauses),
            output_fields=["id"],
            limit=MILVUS_MAX_QUERY_WINDOW,
        )
        return len(rows)

    def existing_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        """For enumerated-set staleness counts: which ids still exist."""
        if not chunk_ids:
            return set()
        client = get_client()
        if not client.has_collection(self.collection):
            return set()
        found: set[str] = set()
        batch = 200
        clean = [str(c) for c in chunk_ids if str(c).strip()]
        for start in range(0, len(clean), batch):
            subset = clean[start : start + batch]
            expr = f"id in [{', '.join(_quote(c) for c in subset)}]"
            rows = client.query(collection_name=self.collection, filter=expr, output_fields=["id"], limit=batch)
            found.update(str(r["id"]) for r in rows)
        return found
