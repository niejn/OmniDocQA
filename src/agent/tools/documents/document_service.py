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

import time
import uuid
from typing import Any

from core.config import config
from loguru import logger

from ..multimodal_asset_store import AssetKeyError, get_asset_store
from ..multimodal_vectorizer import build_vectorizer
from ..rag_stage_log import log_rag, rag_request_scope
from ..retrieval_backends.dense_milvus_multimodal import MilvusMultimodalDenseBackend
from . import document_repository as repo
from .document_repository import DocumentSetError

VALID_KINDS = ("text", "image")

backend = MilvusMultimodalDenseBackend()


class DocumentAskError(Exception):
    def __init__(self, message: str, *, status_code: int = 422, missing: list | None = None) -> None:
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
    collection = (collection or "multimodal").strip().lower()
    if collection not in ("text", "multimodal"):
        raise DocumentAskError(f"collection must be text|multimodal, got {collection!r}")
    top_k = max(1, min(int(top_k or config.document_ask_default_top_k), 50))
    with rag_request_scope(trace_id):
        if collection == "text":
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
            trace_id=trace_id,
            started=started,
        )


async def _ask_multimodal_path(
    *,
    question: str,
    top_k: int,
    generate_answer: bool,
    filters: dict[str, Any] | None,
    set_id: str | None,
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
    try:
        hits = backend.search(
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
    assets = get_asset_store()
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
        image_ref = hit.get("image_ref")
        if image_ref:
            doc_id, _, name = image_ref.partition("/")
            try:
                assets.open(int(doc_id), name)
                item["image_url"] = f"/agent/api/documents/page-image?document_id={doc_id}&name={name}"
            except (KeyError, AssetKeyError, ValueError):
                logger.warning("[Documents] asset missing for chunk {}", hit["chunk_id"])
        evidence.append(item)

    answer = None
    if generate_answer and evidence:
        try:
            answer = await _generate_answer(question, hits)
        except Exception as exc:
            logger.warning("[Documents] generation failed, returning evidence only: {}", exc)
            answer = None
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "trace_id": trace_id,
        "latency_ms": elapsed_ms,
        "collection": "multimodal",
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


def get_collections() -> dict[str, Any]:
    text_backend_available = True
    try:
        from ..milvus_store import get_client

        text_collection = config.milvus_collection
        text_available = get_client().has_collection(text_collection)
    except Exception:
        text_available = False
    try:
        mm_available = backend.has_collection()
        mm_count = backend.count() if mm_available else 0
    except Exception:
        mm_available = False
        mm_count = 0
    return {
        "collections": [
            {
                "id": "text",
                "name": text_collection,
                "available": bool(text_available and text_backend_available),
                "points": None,
            },
            {
                "id": "multimodal",
                "name": config.multimodal_collection,
                "available": bool(mm_available),
                "points": mm_count,
            },
        ]
    }


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
    except Exception as exc:
        raise DocumentAskError(f"create set failed: {exc}", status_code=502) from exc


async def list_sets_with_staleness() -> list[dict[str, Any]]:
    sets = await repo.list_sets()
    for found in sets:
        stale = 0
        if found["kind"] == "enumerated" and found["chunk_ids"]:
            try:
                existing = backend.existing_chunk_ids(found["chunk_ids"])
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
    total = backend.count_document_chunks(int(document_id), kind=kind)
    chunks = backend.query_chunks(
        document_id=int(document_id),
        kind=kind,
        limit=page_size,
        offset=(page - 1) * page_size,
    )
    assets = get_asset_store()
    for chunk in chunks:
        image_ref = chunk.get("image_ref")
        if image_ref:
            doc_id, _, name = image_ref.partition("/")
            try:
                assets.open(int(doc_id), name)
                chunk["image_url"] = f"/agent/api/documents/page-image?document_id={doc_id}&name={name}"
            except (KeyError, AssetKeyError, ValueError):
                chunk.pop("image_ref", None)
    return {
        "document_id": int(document_id),
        "kind": kind,
        "page": page,
        "page_size": page_size,
        "total": total,
        "chunks": chunks,
    }
