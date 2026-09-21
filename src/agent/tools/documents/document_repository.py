"""PG repository for the multimodal document library (T2.1).

Owns two concerns (design §6.1/§6.2/§8):

- ``document_sets`` — user-defined collections, the only NEW table in this
  phase (books stay metadata labels by decision: no independent attributes,
  zero schema churn; ``document_books`` upgrade triggers in §6.1).
- ``rag_documents.metadata`` aggregation — books→chapters filter facet for
  ``GET /documents/filters`` and books→doc_ids expansion for filter
  normalization (the single normalization point lives in document_service).

Set invariants enforced here: unique name (422 on duplicates), enumerated
sets cap at 500 chunk ids (422 above), filter/enumerated kinds are exclusive.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from loguru import logger

from ..node_repository import get_pool

MAX_ENUM_CHUNKS = 500
MAX_SETS = 200

SET_KINDS = ("filter", "enumerated")

# Dynamic-collection registry (§8.5.3 v1): names are Milvus-collection safe
# (lowercase start, lowercase/digit/underscore, 3-32 chars).
COLLECTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")


class DocumentSetError(Exception):
    """Domain error mapped to HTTP 4xx by the API layer."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


async def ensure_sets_table() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_sets (
                set_id      TEXT PRIMARY KEY,
                name        TEXT NOT NULL UNIQUE,
                kind        TEXT NOT NULL,
                filter_json JSONB,
                chunk_ids   JSONB,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )


def _validate_set_payload(name: str, kind: str, filter_json: dict | None, chunk_ids: list[str] | None) -> tuple[str, dict | None, list[str] | None]:
    clean_name = str(name or "").strip()
    if not clean_name or len(clean_name) > 200:
        raise DocumentSetError("set name must be 1-200 chars")
    if kind not in SET_KINDS:
        raise DocumentSetError(f"set kind must be one of {SET_KINDS}")
    if kind == "filter":
        if not filter_json or not isinstance(filter_json, dict):
            raise DocumentSetError("filter sets require a non-empty filter object")
        if not any(filter_json.get(key) for key in ("books", "chapters", "kinds")):
            raise DocumentSetError("filter set requires at least one of books/chapters/kinds")
        return clean_name, filter_json, None
    # enumerated
    clean_ids = [str(c).strip() for c in (chunk_ids or []) if str(c).strip()]
    if not clean_ids:
        raise DocumentSetError("enumerated sets require at least one chunk id")
    if len(clean_ids) > MAX_ENUM_CHUNKS:
        raise DocumentSetError(
            f"enumerated sets cap at {MAX_ENUM_CHUNKS} chunk ids (got {len(clean_ids)}); use a filter set instead"
        )
    return clean_name, None, clean_ids


async def create_set(*, name: str, kind: str, filter_json: dict | None = None, chunk_ids: list[str] | None = None) -> dict[str, Any]:
    clean_name, filter_json, chunk_ids = _validate_set_payload(name, kind, filter_json, chunk_ids)
    pool = await get_pool()
    set_id = str(uuid.uuid4())
    async with pool.acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM document_sets WHERE name = $1", clean_name)
        if exists:
            raise DocumentSetError(f"set name already exists: {clean_name}")
        count = await conn.fetchval("SELECT count(*) FROM document_sets")
        if int(count or 0) >= MAX_SETS:
            raise DocumentSetError(f"set count cap {MAX_SETS} reached; delete unused sets first")
        await conn.execute(
            """
            INSERT INTO document_sets (set_id, name, kind, filter_json, chunk_ids)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
            """,
            set_id,
            clean_name,
            kind,
            json.dumps(filter_json) if filter_json is not None else None,
            json.dumps(chunk_ids) if chunk_ids is not None else None,
        )
    return {"set_id": set_id, "name": clean_name, "kind": kind, "filter_json": filter_json, "chunk_ids": chunk_ids}


def _row_to_set(row: Any) -> dict[str, Any]:
    return {
        "set_id": row["set_id"],
        "name": row["name"],
        "kind": row["kind"],
        "filter_json": json.loads(row["filter_json"]) if row["filter_json"] else None,
        "chunk_ids": json.loads(row["chunk_ids"]) if row["chunk_ids"] else None,
        "created_at": row["created_at"].isoformat(),
    }


def _row_to_collection(row: Any) -> dict[str, Any]:
    return {
        "collection_name": row["collection_name"],
        "embedding_provider": row["embedding_provider"],
        "description": row["description"],
        "created_at": row["created_at"].isoformat(),
    }


async def get_set(set_id: str) -> dict[str, Any] | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM document_sets WHERE set_id = $1", str(set_id))
    return _row_to_set(row) if row is not None else None


async def list_sets() -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM document_sets ORDER BY created_at DESC")
    return [_row_to_set(row) for row in rows]


async def delete_set(set_id: str) -> bool:
    """Delete by id; returns False when the set did not exist (HTTP 404 upstream)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        cmd = await conn.execute("DELETE FROM document_sets WHERE set_id = $1", str(set_id))
    deleted = cmd.split()[-1] if cmd else "0"
    if deleted == "0":
        return False
    logger.info("[DocumentSets] deleted set {}", set_id)
    return True


# ── dynamic-collection registry (§8.5.3 v1) ──────────────────────────────


async def ensure_collections_table() -> None:
    """Idempotent CREATE TABLE for the dynamic-collection registry (ensure_sets_table pattern)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_collections (
                collection_name    TEXT PRIMARY KEY,
                embedding_provider TEXT,
                description        TEXT,
                created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )


def validate_collection_name(name: str) -> str:
    clean = str(name or "").strip()
    if not COLLECTION_NAME_RE.fullmatch(clean):
        raise DocumentSetError(
            "collection name must match ^[a-z][a-z0-9_]{2,31}$ (lowercase start, 3-32 chars)"
        )
    return clean


async def insert_dynamic_collection(
    *, collection_name: str, embedding_provider: str | None, description: str | None
) -> dict[str, Any]:
    """Register one dynamic collection; UniqueViolationError propagates (422 upstream)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO document_collections (collection_name, embedding_provider, description)
            VALUES ($1, $2, $3)
            RETURNING collection_name, embedding_provider, description, created_at
            """,
            collection_name,
            embedding_provider,
            description,
        )
    return _row_to_collection(row)


async def get_dynamic_collection(collection_name: str) -> dict[str, Any] | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT collection_name, embedding_provider, description, created_at "
            "FROM document_collections WHERE collection_name = $1",
            str(collection_name),
        )
    return _row_to_collection(row) if row is not None else None


async def list_dynamic_collections() -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT collection_name, embedding_provider, description, created_at "
            "FROM document_collections ORDER BY created_at DESC"
        )
    return [_row_to_collection(row) for row in rows]


# ── multimodal document inventory (upload API §8.5.2) ────────────────────


async def multimodal_document_exists(document_id: int) -> bool:
    """True when rag_documents holds a row for this id (DELETE /documents/{id} 404 gate)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM rag_documents WHERE id = $1", int(document_id)
        )
    return row is not None


# Placeholder-row lifecycle (review P1-2): uploads RESERVE their document_id by
# inserting this row immediately after allocation, so two concurrent uploads can
# never ingest into the same id. status=ingesting → completed (final upsert) /
# failed (upload failure UPDATE, mid-ingest: row stays for the DELETE cascade) /
# DELETED outright for guard-stage rejections (delete_placeholder_document —
# nothing was written to the stores, so there is nothing to cascade).
PLACEHOLDER_SOURCE = "multimodal_pdf"


async def insert_placeholder_document(
    document_id: int,
    *,
    collection: str,
    filename: str,
    book_meta: dict[str, Any] | None = None,
) -> None:
    """INSERT the upload placeholder row; asyncpg.UniqueViolationError propagates
    (allocate_document_id retries with the next candidate). rag_documents has no
    NOT NULL columns besides id/metadata, so id + metadata are sufficient.
    ``book_meta`` (book_id/chapter_index/chapter_label) is merged in when given
    so even a failed upload leaves a self-describing row instead of a
    numeric-id ghost in the facet (deployment finding #6)."""
    metadata: dict[str, Any] = {
        "source": PLACEHOLDER_SOURCE,
        "status": "ingesting",
        "collection": collection,
        "filename": filename,
    }
    if book_meta:
        metadata.update(
            {
                "book_id": book_meta.get("book_id"),
                "chapter_index": book_meta.get("chapter_index"),
                "chapter_label": book_meta.get("chapter_label"),
            }
        )
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO rag_documents (id, title, source_uri, file_type, metadata)
            VALUES ($1, NULL, NULL, 'multimodal_pdf', $2::jsonb)
            """,
            int(document_id),
            json.dumps(metadata),
        )


async def delete_placeholder_document(document_id: int) -> bool:
    """DELETE a placeholder row outright; True when a row was removed.

    Used when an upload fails in the guard stage (nothing was written to
    Milvus/assets yet) — there is nothing to cascade, so keeping a failed row
    would only pollute the inventory."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        tag = await conn.execute("DELETE FROM rag_documents WHERE id = $1", int(document_id))
    return tag.endswith("DELETE 1")


async def mark_document_ingest_status(document_id: int, status: str) -> bool:
    """Flip the placeholder row's metadata.status (ingesting → failed); False when
    the row is gone (already cleaned up elsewhere)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        cmd = await conn.execute(
            """
            UPDATE rag_documents
            SET metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{status}', to_jsonb($2::text), true),
                updated_at = NOW()
            WHERE id = $1
            """,
            int(document_id),
            str(status),
        )
    return cmd.split()[-1] == "1"


async def multimodal_document_overview() -> list[dict[str, Any]]:
    """Ingested multimodal documents for GET /documents/documents (id DESC).

    Rows are read straight from rag_documents.metadata (written by
    multimodal_ingest.ingest_one_pdf): filename lives in source_uri, title in
    chapter_label, counts in pages/chunks. Placeholder rows mid-upload
    (status=ingesting) or failed uploads (status=failed) are excluded; legacy
    CLI-ingested rows have no status key and show as before (review P1-2).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, title, source_uri, metadata, created_at
            FROM rag_documents
            WHERE metadata->>'source' = 'multimodal_pdf'
              AND COALESCE(metadata->>'status', 'completed') = 'completed'
            ORDER BY id DESC
            """
        )
    return [
        {
            "document_id": int(row["id"]),
            "title": row["title"],
            "source_uri": row["source_uri"],
            "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {}),
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


def build_document_list_item(row: dict[str, Any]) -> dict[str, Any]:
    """Pure mapper overview-row → API item (unit-testable without PG, §8.5.2)."""
    meta = row.get("metadata") or {}
    filename = _source_filename(row.get("source_uri") or "")
    chunks = meta.get("chunks") or {}
    node_count = int(chunks.get("text") or 0) + int(chunks.get("image") or 0)
    chapter_label = str(meta.get("chapter_label") or row.get("title") or filename)
    return {
        "document_id": int(row["document_id"]),
        "filename": str(meta.get("filename") or filename),
        "title": str(meta.get("chapter_label") or row.get("title") or filename),
        "page_count": int(meta.get("pages") or 0),
        "node_count": int(meta.get("node_count") or node_count),
        "book_id": str(meta.get("book_id") or row["document_id"]),
        "chapter_label": chapter_label,
        "created_at": str(row.get("created_at") or ""),
    }


# ── rag_documents.metadata aggregation (filters facet + books expansion) ──


async def list_multimodal_documents() -> list[dict[str, Any]]:
    """Completed multimodal rows with hierarchy metadata (drives GET /filters).

    Same visibility rule as ``multimodal_document_overview``: placeholder rows
    mid-upload (status=ingesting) and failed uploads (status=failed) are
    excluded — otherwise every quota/rejection failure would surface as a
    bogus 《<document_id>》 book in the facet (deployment finding #6).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, title, source_uri, metadata
            FROM rag_documents
            WHERE metadata->>'source' = 'multimodal_pdf'
              AND COALESCE(metadata->>'status', 'completed') = 'completed'
            ORDER BY (metadata->>'book_id'), (metadata->>'chapter_index')::int NULLS LAST, id
            """
        )
    return [
        {
            "document_id": int(row["id"]),
            "title": row["title"],
            "source_uri": row["source_uri"],
            "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {}),
        }
        for row in rows
    ]


def aggregate_filters(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Group multimodal rows into the ``GET /documents/filters`` facet payload.

    Pure function over ``list_multimodal_documents`` output so it stays unit
    testable without PG: books in insertion order, chapters by chapter_index,
    chunk counts read straight from metadata (lazy — no Milvus roundtrip).
    """
    books: dict[str, dict[str, Any]] = {}
    kinds_seen: set[str] = set()
    for doc in documents:
        meta = doc.get("metadata") or {}
        book_id = str(meta.get("book_id") or doc["document_id"])
        chunks = meta.get("chunks") or {}
        kinds_seen.update(k for k in ("text", "image") if (chunks or {}).get(k))
        chapter = {
            "document_id": doc["document_id"],
            "chapter_index": int(meta.get("chapter_index") or 1),
            "chapter_label": meta.get("chapter_label") or doc.get("title") or str(doc["document_id"]),
            "filename": _source_filename(doc.get("source_uri") or ""),
            "pages": int(meta.get("pages") or 0),
            "chunks": {
                "text": int((chunks or {}).get("text") or 0),
                "image": int((chunks or {}).get("image") or 0),
            },
        }
        book = books.setdefault(
            book_id,
            {"book_id": book_id, "chapter_count": 0, "chunk_count": 0, "chapters": []},
        )
        book["chapters"].append(chapter)
        book["chapter_count"] += 1
        book["chunk_count"] += chapter["chunks"]["text"] + chapter["chunks"]["image"]
    book_list = sorted(books.values(), key=lambda b: b["book_id"])
    for book in book_list:
        book["chapters"].sort(key=lambda c: c["chapter_index"])
    return {"books": book_list, "kinds": sorted(kinds_seen)}


def _source_filename(source_uri: str) -> str:
    from pathlib import Path

    try:
        return Path(source_uri).name
    except Exception:
        return source_uri or ""


async def expand_books_to_doc_ids(book_ids: list[str]) -> tuple[list[int], list[str]]:
    """books → document_id union; returns (found_doc_ids, missing_book_ids)."""
    wanted = {str(b).strip() for b in book_ids if str(b).strip()}
    if not wanted:
        return [], []
    documents = await list_multimodal_documents()
    found: list[int] = []
    seen_books: set[str] = set()
    for doc in documents:
        meta = doc.get("metadata") or {}
        book_id = str(meta.get("book_id") or "")
        if book_id in wanted:
            seen_books.add(book_id)
            found.append(int(doc["document_id"]))
    missing = sorted(wanted - seen_books)
    return sorted(set(found)), missing


async def expand_chapters(document_ids: list[int]) -> tuple[list[int], list[int]]:
    """chapters (=document_ids) validation; returns (valid_ids, missing_ids)."""
    wanted = sorted({int(d) for d in document_ids})
    if not wanted:
        return [], []
    known = {doc["document_id"] for doc in await list_multimodal_documents()}
    valid = [d for d in wanted if d in known]
    missing = [d for d in wanted if d not in known]
    return valid, missing
