"""Multimodal document library service (T2.3) — the filter normalization point.

Two retrieval paths behind ``POST /documents/ask`` (§8):

- ``collection="multimodal"``: question → multimodal embedding → dense top-k
  over ``rag_multimodal`` with a single pushed-down Milvus expression
  (document-id union from books∪chapters, kind list, or enumerated-set PKs);
  optional generation over the text of the hits (LLM never sees images —
  image chunks contribute their VLM description, §3 image three-stage roles).
- ``collection="text"``: the existing hybrid retrieval over ``rag_nodes``
  (full library scope), rerank included; ``filters`` are ignored with a
  warning log (zero-change proof for /ask semantics, §10-9).

Filter normalization (§6.1, single point by design): books expand via PG
metadata to document ids, chapters ARE document ids; the two are UNIONed and
deduplicated (no book∩chapter intersection trap). Explicit filters that
normalize to an EMPTY id set are a 422 — typos must not silently widen to the
whole library. ``filters`` and ``set_id`` are mutually exclusive (422).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
from core.config import config
from loguru import logger

from ..milvus_store import MILVUS_MAX_QUERY_WINDOW
from ..multimodal_asset_store import AssetKeyError, get_asset_store
from ..multimodal_vectorizer import build_vectorizer
from ..rag_stage_log import log_rag, rag_request_scope
from ..retrieval_backends.dense_milvus_multimodal import MilvusMultimodalDenseBackend
from . import document_repository as repo
from .document_repository import (
    DocumentSetError,
    build_document_list_item,
    validate_collection_name,
)

VALID_KINDS = ("text", "image")

backend = MilvusMultimodalDenseBackend()


class DocumentAskError(Exception):
    def __init__(self, message: str, *, status_code: int = 422, missing: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.missing = missing


async def normalize_filters(
    filters: dict[str, Any] | None,
    set_id: str | None,
) -> dict[str, Any]:
    """Normalize filters/set_id into backend search kwargs (single point, §8).

    Returns ``{"document_ids": [...], "kinds": [...], "chunk_ids": [...]}``.
    Raises DocumentAskError(422/404) per the error matrix (§15).
    """
    if filters and set_id:
        raise DocumentAskError("filters and set_id are mutually exclusive; pass one")
    if set_id:
        found = await repo.get_set(set_id)
        if found is None:
            raise DocumentAskError(f"set not found: {set_id}", status_code=404)
        if found["kind"] == "enumerated":
            return {"document_ids": None, "kinds": None, "chunk_ids": list(found["chunk_ids"] or [])}
        filters = found["filter_json"]

    if not filters:
        return {"document_ids": None, "kinds": None, "chunk_ids": None}

    if not isinstance(filters, dict):
        raise DocumentAskError("filters must be an object")

    kinds = filters.get("kinds") or None
    if kinds is not None:
        kinds = [str(k).strip().lower() for k in kinds if str(k).strip()]
        bad = [k for k in kinds if k not in VALID_KINDS]
        if bad:
            raise DocumentAskError(f"invalid kinds: {bad}; allowed {list(VALID_KINDS)}")
        kinds = kinds or None

    books = [str(b).strip() for b in (filters.get("books") or []) if str(b).strip()]
    chapters = [int(c) for c in (filters.get("chapters") or [])]

    doc_ids: set[int] = set()
    missing_books: list[str] = []
    missing_chapters: list[int] = []
    if books:
        found_ids, missing_books = await repo.expand_books_to_doc_ids(books)
        doc_ids.update(found_ids)
    if chapters:
        valid_ids, missing_chapters = await repo.expand_chapters(chapters)
        doc_ids.update(valid_ids)
    explicit = bool(books or chapters)
    if explicit and not doc_ids:
        raise DocumentAskError(
            "filters matched no documents (typo guard): books/chapters unknown",
            missing={"books": missing_books, "chapters": missing_chapters},
        )
    log_rag(
        "mm_filter_normalize",
        books=books or None,
        chapters=chapters or None,
        missing_books=missing_books or None,
        missing_chapters=missing_chapters or None,
        doc_union=sorted(doc_ids) or None,
        kinds=kinds,
    )
    return {"document_ids": sorted(doc_ids) or None, "kinds": kinds, "chunk_ids": None}


async def _generate_answer(question: str, hits: list[dict[str, Any]]) -> str:
    """Generation over hit text (image chunks contribute descriptions only)."""
    from tools.llm import get_llm

    budget = max(1000, int(config.context_char_budget or 6000))
    context_parts: list[str] = []
    used = 0
    for i, hit in enumerate(hits, 1):
        label = f"[{i}] {hit.get('filename') or ''} 第{(hit.get('page_no') or 0) + 1}页"
        title = hit.get("title")
        if title:
            label += f" · {title}"
        body = (hit.get("text_preview") or "").strip()
        if hit.get("kind") == "image":
            body = f"（图片描述）{body}"
        part = f"{label}\n{body}"
        if used + len(part) > budget:
            break
        context_parts.append(part)
        used += len(part)
    prompt = (
        "你是多模态文档库问答助手。仅依据以下检索到的文档片段回答问题；"
        "片段中的图片以文字描述形式提供。回答需给出结论并标注来源编号（如 [1]）；"
        "片段不足以回答时明确说明。\n\n"
        f"问题：{question}\n\n文档片段：\n" + "\n\n".join(context_parts)
    )
    llm = get_llm(temperature=0.3)
    response = await llm.ainvoke(prompt)
    return (response.content or "").strip()


async def resolve_multimodal_collection(value: str | None) -> str:
    """Map an ask/upload collection value onto a concrete Milvus collection.

    Accepts the fixed alias ``multimodal`` (→ ``config.multimodal_collection``)
    and registered dynamic collections (§8.5.3). Anything else is a 422 — a
    typo must not silently fall back to the default library.
    """
    raw = (value or "").strip()
    if not raw or raw.lower() == "multimodal" or raw == config.multimodal_collection:
        return config.multimodal_collection
    found = await repo.get_dynamic_collection(raw)
    if found is not None:
        return str(found["collection_name"])
    raise DocumentAskError(
        f"unknown collection: {raw!r}; pass 'multimodal' or a registered dynamic collection"
    )


async def resolve_upload_collection(value: str | None) -> str:
    """Upload-target validation (§8.5.2): one of the two fixed collections or a registry entry.

    The SEC text collection (``rag_nodes``) is a 422 — multimodal ingest writes
    its own Milvus schema, and a text-library upload would poison the shared
    text collection (review P2-2).
    """
    raw = (value or "").strip()
    if not raw or raw.lower() == "multimodal" or raw == config.multimodal_collection:
        return config.multimodal_collection
    if raw == config.milvus_collection:
        raise DocumentAskError(
            f"多模态上传仅支持多模态 collection: {config.milvus_collection!r} 是 SEC 文本库 (rag_nodes)"
        )
    return await resolve_multimodal_collection(raw)


async def ask_documents(
    *,
    question: str,
    collection: str = "multimodal",
    top_k: int | None = None,
    generate_answer: bool = False,
    filters: dict[str, Any] | None = None,
    set_id: str | None = None,
    document_ids_text: list[int] | None = None,
) -> dict[str, Any]:
    """Entry for POST /documents/ask. Returns the ask response payload."""
    started = time.perf_counter()
    trace_id = f"docs-{uuid.uuid4().hex[:12]}"
    raw_collection = (collection or "multimodal").strip()
    is_text_path = raw_collection.lower() == "text"
    mm_collection = None if is_text_path else await resolve_multimodal_collection(raw_collection)
    top_k = max(1, min(int(top_k or config.document_ask_default_top_k), 50))
    with rag_request_scope(trace_id):
        if is_text_path:
            if filters or set_id:
                logger.warning(
                    "[Documents] text path ignores filters/set_id (zero-change boundary)"
                )
            return await _ask_text_path(
                question=question,
                top_k=top_k,
                generate_answer=generate_answer,
                document_ids_text=document_ids_text,
                trace_id=trace_id,
                started=started,
            )
        return await _ask_multimodal_path(
            question=question,
            top_k=top_k,
            generate_answer=generate_answer,
            filters=filters,
            set_id=set_id,
            milvus_collection=mm_collection,
            trace_id=trace_id,
            started=started,
        )


def _attach_image_urls(items: list[dict[str, Any]]) -> None:
    """Sync asset-existence probe per evidence item (runs in a worker thread).

    asset-missing degradation (§15): drop image_url, keep the text evidence.
    """
    assets = get_asset_store()
    for item in items:
        image_ref = item.get("image_ref")
        if not image_ref:
            continue
        doc_id, _, name = str(image_ref).partition("/")
        try:
            assets.open(int(doc_id), name)
            item["image_url"] = f"/agent/api/documents/page-image?document_id={doc_id}&name={name}"
        except (KeyError, AssetKeyError, ValueError):
            logger.warning("[Documents] asset missing for chunk {}", item.get("chunk_id"))


async def _ask_multimodal_path(
    *,
    question: str,
    top_k: int,
    generate_answer: bool,
    filters: dict[str, Any] | None,
    set_id: str | None,
    milvus_collection: str | None = None,
    trace_id: str,
    started: float,
) -> dict[str, Any]:
    normalized = await normalize_filters(filters, set_id)
    if not question or not str(question).strip():
        raise DocumentAskError("question is required")
    vectorizer = build_vectorizer()
    try:
        embed = await vectorizer.embed_text(str(question).strip())
    finally:
        await vectorizer.aclose()
    if embed.vector is None:
        log_rag("mm_ask_error", level="error", error=embed.error or "embedding failed")
        raise DocumentAskError(f"question embedding failed: {embed.error}", status_code=502)
    # One backend per request: dynamic collections (§8.5.3) get their own
    # MilvusMultimodalDenseBackend; None = module default (config collection).
    backend_for_request = (
        backend if not milvus_collection or milvus_collection == backend.collection
        else MilvusMultimodalDenseBackend(collection=milvus_collection)
    )
    try:
        # Sync pymilvus search → worker thread so the event loop never blocks (review A1).
        hits = await asyncio.to_thread(
            backend_for_request.search,
            embed.vector,
            document_ids=normalized["document_ids"],
            kinds=normalized["kinds"],
            chunk_ids=normalized["chunk_ids"],
            limit=top_k,
            log_stage="mm_search",
        )
    except ValueError:
        raise
    except Exception as exc:
        log_rag("mm_ask_error", level="error", error=str(exc)[:300])
        raise DocumentAskError(f"multimodal search failed: {exc}", status_code=502) from exc

    # asset-missing degradation (§15): drop image_url, keep the text evidence.
    evidence: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, 1):
        item: dict[str, Any] = {
            "rank": rank,
            "chunk_id": hit["chunk_id"],
            "kind": hit["kind"],
            "score": round(hit["score"], 6),
            "document_id": hit["document_id"],
            "filename": hit["filename"],
            "title": hit["title"],
            "page_no": hit["page_no"],
            "text_preview": hit["text_preview"],
            "book_id": hit["book_id"],
            "chapter_label": hit["chapter_label"],
        }
        evidence.append(item)
    await asyncio.to_thread(_attach_image_urls, evidence)

    answer = None
    if generate_answer and evidence:
        try:
            answer = await _generate_answer(question, hits)
        except Exception as exc:
            logger.warning("[Documents] generation failed, returning evidence only: {}", exc)
            answer = None
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    resolved_label = (
        "multimodal"
        if not milvus_collection or milvus_collection == config.multimodal_collection
        else milvus_collection
    )
    return {
        "trace_id": trace_id,
        "latency_ms": elapsed_ms,
        "collection": resolved_label,
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "counts": {
            "candidates": len(evidence),
            "with_image": sum(1 for e in evidence if e.get("image_url")),
            "top_k": top_k,
        },
    }


async def _ask_text_path(
    *,
    question: str,
    top_k: int,
    generate_answer: bool,
    document_ids_text: list[int] | None,
    trace_id: str,
    started: float,
) -> dict[str, Any]:
    """Full-library hybrid retrieval over rag_nodes + optional generation."""
    from tools.llamaindex_retrieval import retrieval_service
    from tools.node_repository import list_available_document_ids

    if not question or not str(question).strip():
        raise DocumentAskError("question is required")
    doc_ids = document_ids_text or await list_available_document_ids(limit=5000)
    doc_ids = [d for d in doc_ids if d not in config.rag_ask_excluded_document_id_set]
    if not doc_ids:
        return {
            "trace_id": trace_id,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "collection": "text",
            "question": question,
            "answer": None,
            "evidence": [],
            "counts": {"candidates": 0, "top_k": top_k},
        }
    try:
        result = await retrieval_service.retrieve(query=str(question).strip(), document_ids=doc_ids)
    except Exception as exc:
        log_rag("text_ask_error", level="error", error=str(exc)[:300])
        raise DocumentAskError(f"text retrieval failed: {exc}", status_code=502) from exc
    nodes = result.get("nodes") or []
    evidence = [
        {
            "rank": rank,
            "node_id": node["node_id"],
            "kind": "text",
            "score": round(float(node.get("score") or 0.0), 6),
            "document_id": node.get("document_id"),
            "title": node.get("title"),
            "text_preview": (node.get("text") or "")[:2000],
        }
        for rank, node in enumerate(nodes[:top_k], 1)
    ]
    answer = None
    if generate_answer and evidence:
        try:
            answer = await _generate_answer(question, evidence)
        except Exception as exc:
            logger.warning("[Documents] text generation failed: {}", exc)
    return {
        "trace_id": trace_id,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "collection": "text",
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "counts": {"candidates": len(evidence), "top_k": top_k},
    }


# ── collections / page-image / sets / chapters (MM-4) ────────────────────


async def get_collections() -> dict[str, Any]:
    text_collection = config.milvus_collection

    def _probe_text() -> bool:
        try:
            from ..milvus_store import get_client

            return bool(get_client().has_collection(text_collection))
        except Exception:
            return False

    def _probe_mm() -> tuple[bool, int]:
        try:
            available = backend.has_collection()
            return available, backend.count() if available else 0
        except Exception:
            return False, 0

    # Sync pymilvus probes → worker thread (review A1).
    text_available, (mm_available, mm_count) = await asyncio.gather(
        asyncio.to_thread(_probe_text), asyncio.to_thread(_probe_mm)
    )
    entries: list[dict[str, Any]] = [
        {
            "id": "text",
            "name": text_collection,
            "available": bool(text_available),
            "points": None,
            "kind": "fixed",
        },
        {
            "id": "multimodal",
            "name": config.multimodal_collection,
            "available": bool(mm_available),
            "points": mm_count,
            "kind": "fixed",
        },
    ]
    # Dynamic registry entries (§8.5.3): availability/points probed per collection.
    try:
        dynamic = await repo.list_dynamic_collections()
    except Exception:
        dynamic = []

    def _probe_dynamic(target: str) -> tuple[bool, int | None]:
        probe = MilvusMultimodalDenseBackend(collection=target)
        try:
            if not probe.has_collection():
                return False, None
            return True, probe.count()
        except Exception:
            return False, None

    # Sync pymilvus probes → concurrent worker threads (review P2-8: serial
    # round-trips made the endpoint latency grow with the registry size).
    probe_results = await asyncio.gather(
        *(asyncio.to_thread(_probe_dynamic, str(entry["collection_name"])) for entry in dynamic)
    )
    for entry, (available, points) in zip(dynamic, probe_results):
        name = str(entry["collection_name"])
        entries.append(
            {
                "id": name,
                "name": name,
                "available": bool(available),
                "points": points,
                "kind": "dynamic",
            }
        )
    return {"collections": entries}


def get_page_image(document_id: int, name: str) -> bytes:
    try:
        return get_asset_store().open(int(document_id), str(name))
    except AssetKeyError as exc:
        raise DocumentAskError(str(exc), status_code=400) from exc
    except KeyError as exc:
        raise DocumentAskError(f"image not found: {document_id}/{name}", status_code=404) from exc


async def get_filters() -> dict[str, Any]:
    documents = await repo.list_multimodal_documents()
    return repo.aggregate_filters(documents)


async def create_set(name: str, filter_json: dict | None, chunk_ids: list[str] | None) -> dict[str, Any]:
    kind = "enumerated" if chunk_ids is not None else "filter"
    try:
        return await repo.create_set(name=name, kind=kind, filter_json=filter_json, chunk_ids=chunk_ids)
    except DocumentSetError as exc:
        # Repository domain errors (duplicate name / cap exceeded) map to 422.
        raise DocumentAskError(str(exc), status_code=exc.status_code) from exc
    except asyncpg.UniqueViolationError as exc:
        # Concurrent duplicate insert lost the pre-check race → client error, not 502.
        raise DocumentAskError(f"set name already exists (concurrent insert): {name}") from exc
    except Exception as exc:
        raise DocumentAskError(f"create set failed: {exc}", status_code=502) from exc


async def list_sets_with_staleness() -> list[dict[str, Any]]:
    sets = await repo.list_sets()
    for found in sets:
        stale = 0
        if found["kind"] == "enumerated" and found["chunk_ids"]:
            try:
                # Sync pymilvus query → worker thread (review A1).
                existing = await asyncio.to_thread(backend.existing_chunk_ids, found["chunk_ids"])
                stale = sum(1 for c in found["chunk_ids"] if c not in existing)
            except Exception:
                stale = 0
        found["stale_chunk_count"] = stale
        if found["kind"] == "enumerated":
            found["chunk_count"] = len(found["chunk_ids"] or [])
    return sets


async def delete_set(set_id: str) -> None:
    deleted = await repo.delete_set(set_id)
    if not deleted:
        raise DocumentAskError(f"set not found: {set_id}", status_code=404)


async def get_chapter_chunks(
    document_id: int,
    *,
    kind: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """MM-4 chapter browsing: paginated chunks of one chapter (=document)."""
    if kind is not None and kind not in VALID_KINDS:
        raise DocumentAskError(f"invalid kind: {kind}; allowed {list(VALID_KINDS)}")
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 100))
    offset = (page - 1) * page_size
    if offset + page_size > MILVUS_MAX_QUERY_WINDOW:
        # Milvus rejects offset+limit beyond the query window — a client-sized
        # page, so 422 not 502.
        raise DocumentAskError(
            f"page {page} x page_size {page_size} exceeds the Milvus query window "
            f"(offset+limit must be <= {MILVUS_MAX_QUERY_WINDOW})"
        )
    total = await asyncio.to_thread(backend.count_document_chunks, int(document_id), kind)
    chunks = await asyncio.to_thread(
        backend.query_chunks,
        document_id=int(document_id),
        kind=kind,
        limit=page_size,
        offset=offset,
    )
    await asyncio.to_thread(_attach_image_urls, chunks)
    return {
        "document_id": int(document_id),
        "kind": kind,
        "page": page,
        "page_size": page_size,
        "total": total,
        "chunks": chunks,
    }


# ── upload / inventory / delete (§8.5.2) + dynamic collections (§8.5.3) ──


# Bounded retry budget for the placeholder-row reservation race (review P1-2).
_MAX_ALLOCATE_ATTEMPTS = 200


async def allocate_document_id(
    milvus_collection: str, *, filename: str, book_meta: dict[str, Any] | None = None
) -> int:
    """Auto-allocate and RESERVE a collision-free document_id for an upload.

    node_repository has no allocator (ids are caller-assigned, e.g.
    --document-id-start), so uploads take MAX(rag_documents.id)+1, bumping past
    any id that already holds points in the target Milvus collection. The
    reservation IS the placeholder INSERT (status=ingesting, review P1-2):
    MAX+1 alone raced with the whole ingest duration, letting two concurrent
    uploads pick the same id and overwrite each other. A loser of the INSERT
    race (UniqueViolationError) bumps its candidate and retries, bounded.
    ``book_meta`` goes onto the placeholder so a failed upload stays
    self-describing (facet shows the file, not a numeric-id ghost).
    """
    from ..node_repository import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COALESCE(MAX(id), 0) AS max_id FROM rag_documents")
    candidate = int(row["max_id"] or 0) + 1
    for _ in range(_MAX_ALLOCATE_ATTEMPTS):
        while await asyncio.to_thread(_milvus_document_exists, candidate, milvus_collection):
            candidate += 1
        try:
            await repo.insert_placeholder_document(
                candidate, collection=milvus_collection, filename=filename, book_meta=book_meta
            )
        except asyncpg.UniqueViolationError:
            candidate += 1  # concurrent upload won this id; take the next
            continue
        return candidate
    raise DocumentAskError(
        f"could not reserve a free document_id after {_MAX_ALLOCATE_ATTEMPTS} attempts",
        status_code=502,
    )


def _milvus_document_exists(document_id: int, collection: str) -> bool:
    from ..milvus_store import get_client

    client = get_client()
    if not client.has_collection(collection):
        return False
    rows = client.query(
        collection_name=collection,
        filter=f"document_id == {int(document_id)}",
        output_fields=["id"],
        limit=1,
    )
    return bool(rows)


async def upload_multimodal_pdf(*, pdf_path: Path, collection: str | None = None) -> dict[str, Any]:
    """Six-step ingest of one uploaded PDF (shared core: multimodal_ingest.ingest_one_pdf).

    ``source`` in extra_metadata must equal multimodal_cleanup._detect's marker
    (``metadata->>'source' = 'multimodal_pdf'``) so DELETE routes the cascade
    to the right stores; ``collection`` records the target for the same reason.
    Returns the UploadResult payload {document_id, status, node_count, page_count, filename}.

    Placeholder-row state machine (review P1-2): allocate_document_id INSERTs
    the row with status=ingesting; the final upsert merges extra_metadata
    (status=completed) over it — ``ON CONFLICT (id)`` does a jsonb ``||`` merge,
    and every placeholder key (source/status/collection/filename) is also
    carried by the final metadata, so the merge is a full overwrite of the
    placeholder's view. On ANY ingest failure the placeholder is flagged
    status=failed via UPDATE (the row stays so the DELETE cascade can clean it)
    and the exception propagates for 422/502 classification.
    """
    from ..multimodal_ingest import UploadRejectedError, build_book_meta, ingest_one_pdf

    target = await resolve_upload_collection(collection)
    pdf_path = Path(pdf_path)
    filename = pdf_path.name
    (book_meta,) = build_book_meta(None, 1, [pdf_path])
    document_id = await allocate_document_id(target, filename=filename, book_meta=book_meta)
    extra_metadata = {
        "source": "multimodal_pdf",
        "collection": target,
        "status": "completed",
        "uploaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    try:
        # replace=True: the placeholder row is ours (guard passes) and a fresh
        # id has no old Milvus points to delete.
        summary = await ingest_one_pdf(
            pdf_path,
            document_id,
            book_meta,
            collection=target,
            extra_metadata=extra_metadata,
            replace=True,
        )
    except asyncio.CancelledError:
        # Browser disconnect / page refresh mid-upload cancels the request task
        # (uvicorn cancels the handler). Nothing is still writing at this point
        # by definition — DELETE the placeholder so the inventory doesn't keep
        # an eternal 入库中 ghost; cancellation then propagates to uvicorn.
        try:
            await repo.delete_placeholder_document(document_id)
        except Exception as del_exc:
            logger.warning(
                "[Documents] cancelled upload {}: placeholder delete failed ({})",
                document_id,
                del_exc,
            )
        raise
    except UploadRejectedError:
        # Guard-stage rejection: the guards run before ANY store write, so
        # there is nothing to cascade — DELETE the placeholder outright
        # instead of leaving a failed ghost in the inventory (deployment
        # finding #6: quota-failed uploads piled up as 《<id>》 facet junk).
        try:
            await repo.delete_placeholder_document(document_id)
        except Exception as del_exc:
            logger.warning(
                "[Documents] failed to delete rejected placeholder {}: {}", document_id, del_exc
            )
        raise
    except Exception:
        # Mid-ingest failure: Milvus points / assets may exist, so the row
        # stays (status=failed) for the DELETE cascade to clean up.
        try:
            await repo.mark_document_ingest_status(document_id, "failed")
        except Exception as mark_exc:
            logger.warning(
                "[Documents] failed to flag placeholder {} as failed: {}", document_id, mark_exc
            )
        raise
    return {
        "document_id": int(document_id),
        "status": "completed",
        "node_count": int(summary.get("chunks_text") or 0) + int(summary.get("chunks_image") or 0),
        "page_count": int(summary.get("pages") or 0),
        "filename": str(summary.get("filename") or pdf_path.name),
    }


def classify_upload_failure(exc: Exception) -> int:
    """Map an ingest_one_pdf failure to 422 / 502 / 503.

    Decided by EXCEPTION TYPE, not message substrings (review P2-1):
    UploadRejectedError (the ingest guards' ValueError subclass: oversize,
    page cap, unguarded overwrite) → 422; EmbeddingQuotaExceededError (ark
    AccountQuotaExceeded — service-side quota, retrying cannot fix it until
    the provider resets it) → 503; everything else — including the
    dim/collection/embedding ValueError family and RuntimeError/Exception —
    → 502.
    """
    from ..multimodal_ingest import UploadRejectedError
    from ..multimodal_vectorizer import EmbeddingQuotaExceededError

    if isinstance(exc, EmbeddingQuotaExceededError):
        return 503
    return 422 if isinstance(exc, UploadRejectedError) else 502


async def list_documents() -> list[dict[str, Any]]:
    """GET /documents/documents payload: bare array, document_id DESC (§8.5.2)."""
    rows = await repo.multimodal_document_overview()
    return [build_document_list_item(row) for row in rows]


async def delete_document(document_id: int) -> dict[str, Any]:
    """DELETE /documents/{id}: 404 when unknown, else the shared deletion cascade."""
    from ..multimodal_cleanup import delete_document_everywhere

    if not await repo.multimodal_document_exists(document_id):
        raise DocumentAskError(f"document not found: {document_id}", status_code=404)
    return await delete_document_everywhere(int(document_id))


async def create_dynamic_collection(
    *,
    name: str,
    embedding_provider: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """POST /collections (§8.5.3): validate → create Milvus collection → register.

    Milvus failures raise 502 and leave the registry untouched (no phantom
    entries); duplicate names (fixed or registered) are 422.
    """
    try:
        clean = validate_collection_name(name)
    except DocumentSetError as exc:
        raise DocumentAskError(str(exc)) from exc
    reserved = {"text", "multimodal", config.milvus_collection, config.multimodal_collection}
    if clean in reserved:
        raise DocumentAskError(f"collection name conflicts with a fixed collection: {clean!r}")
    if await repo.get_dynamic_collection(clean) is not None:
        raise DocumentAskError(f"collection already registered: {clean!r}")
    dim = int(config.multimodal_embedding_dim or 0)
    if not dim:
        raise DocumentAskError(
            "MULTIMODAL_EMBEDDING_DIM is unset (0 = probe on first embed); "
            "set it explicitly to pre-create a collection"
        )
    try:
        # Module-level backend class (patchable in tests); sync pymilvus → thread.
        await asyncio.to_thread(MilvusMultimodalDenseBackend(collection=clean).ensure_collection, dim)
    except Exception as exc:
        raise DocumentAskError(
            f"Milvus collection creation failed for {clean!r}: {exc}", status_code=502
        ) from exc
    provider = (embedding_provider or config.multimodal_embedding_provider or "").strip() or None
    try:
        return await repo.insert_dynamic_collection(
            collection_name=clean, embedding_provider=provider, description=(description or "").strip() or None
        )
    except asyncpg.UniqueViolationError as exc:
        raise DocumentAskError(f"collection already registered (concurrent insert): {clean!r}") from exc


# ── §8.5.4 testset / evaluate jobs ────────────────────────────────────────
# v1 simplification: jobs live in an in-process dict guarded by an asyncio
# lock — state is LOST ON RESTART and not shared across workers. Job bodies
# catch every Exception so a failing task records the error instead of
# crashing the process.

AGENT_ROOT = Path(__file__).resolve().parents[2]

JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = asyncio.Lock()
_JOBS_MAX = 50  # evict oldest completed/failed beyond this; running jobs untouched
_EVAL_TOP_K = 8  # mirrors run_multimodal_eval's CLI default


def eval_whitelist_dir() -> Path:
    """Absolute whitelist dir for generated testsets / reports (MULTIMODAL_EVAL_DIR)."""
    raw = Path(config.multimodal_eval_dir)
    return raw if raw.is_absolute() else (AGENT_ROOT / raw)


def resolve_testset_path(raw: str) -> Path:
    """Whitelist-resolve ``testset_path`` (blocks arbitrary-path reads/writes).

    Relative paths resolve from the package root (src/agent), matching the
    scripts' relative default paths. Non-whitelisted / missing files → 422.
    """
    candidate = Path(str(raw or "")).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (AGENT_ROOT / candidate).resolve()
    whitelist = eval_whitelist_dir().resolve()
    if whitelist != resolved and whitelist not in resolved.parents:
        raise DocumentAskError(
            f"testset_path must be inside the whitelist directory {whitelist}",
        )
    if not resolved.is_file():
        raise DocumentAskError(f"testset file not found: {resolved}")
    return resolved


def get_job(job_id: str) -> dict[str, Any]:
    """Public job view (task handle stripped); 404 when unknown."""
    job = JOBS.get(str(job_id))
    if job is None:
        raise DocumentAskError(f"job not found: {job_id}", status_code=404)
    return {key: value for key, value in job.items() if key != "task"}


async def _mark_job(job_id: str, **fields: Any) -> None:
    async with _JOBS_LOCK:
        JOBS[job_id].update(fields)


async def _finish_job_error(job_id: str, exc: Exception) -> None:
    logger.exception("[Documents] job {} failed: {}", job_id, exc)
    await _mark_job(job_id, status="failed", error=str(exc)[:500])


async def _register_job(kind: str) -> str:
    """Create a job entry under the lock, evicting the OLDEST finished job when
    the in-process dict hits _JOBS_MAX (running jobs are never evicted; if every
    entry is still running the dict is allowed past the cap rather than killing
    work — review P2-7)."""
    async with _JOBS_LOCK:
        if len(JOBS) >= _JOBS_MAX:
            for jid, job in sorted(JOBS.items(), key=lambda kv: str(kv[1].get("created_at") or "")):
                if job.get("status") in ("completed", "failed"):
                    del JOBS[jid]
                    break
        job_id = uuid.uuid4().hex[:16]
        JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        return job_id


async def start_testset_job(*, collection: str | None, testset_size: int) -> str:
    """Spawn the background testset generation; returns the job_id immediately."""
    job_id = await _register_job("testset")
    task = asyncio.create_task(_run_testset_job(job_id, collection, int(testset_size)))
    async with _JOBS_LOCK:
        JOBS[job_id]["task"] = task  # keep a ref so the task cannot be GC'd mid-run
    return job_id


async def _run_testset_job(job_id: str, collection: str | None, testset_size: int) -> None:
    from scripts.gen_multimodal_evalset import generate_evalset_core

    try:
        payload = await generate_evalset_core(collection=collection, total=testset_size)
        out_path = eval_whitelist_dir() / f"testset_{job_id}.json"
        body = json.dumps(payload, ensure_ascii=False, indent=2)

        def _write() -> None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(body, encoding="utf-8")

        await asyncio.to_thread(_write)
        await _mark_job(
            job_id,
            status="completed",
            questions_count=len(payload.get("questions") or []),
            questions=payload.get("questions") or [],
            testset_path=str(out_path),
        )
    except Exception as exc:  # noqa: BLE001 — jobs must never crash the process
        await _finish_job_error(job_id, exc)


async def start_evaluation_job(*, collection: str | None, testset_path: Path) -> str:
    """Spawn the background gold-set evaluation; returns the job_id immediately."""
    job_id = await _register_job("evaluate")
    task = asyncio.create_task(_run_evaluation_job(job_id, collection, testset_path))
    async with _JOBS_LOCK:
        JOBS[job_id]["task"] = task
    return job_id


async def _run_evaluation_job(job_id: str, collection: str | None, testset_path: Path) -> None:
    from scripts.run_multimodal_eval import run_evaluation_core

    try:
        report = await run_evaluation_core(
            evalset=testset_path,
            top_k=_EVAL_TOP_K,
            collection=collection or None,
            progress=lambda _msg: None,
        )
        report_path = eval_whitelist_dir() / "reports" / f"mm_eval_api_{job_id}.json"
        body = json.dumps(report, ensure_ascii=False, indent=2)

        def _write() -> None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(body, encoding="utf-8")

        await asyncio.to_thread(_write)
        await _mark_job(job_id, status="completed", report=report, report_path=str(report_path))
    except Exception as exc:  # noqa: BLE001 — jobs must never crash the process
        await _finish_job_error(job_id, exc)
