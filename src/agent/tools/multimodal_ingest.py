"""Multimodal PDF ingest core (§8.5.2) — the six-step pipeline, importable.

Extracted verbatim from ``scripts/ingest_multimodal_pdf.py`` so both the CLI
and ``POST /agent/api/documents/upload`` share one pipeline (§8.5.2: the
upload endpoint runs the same six steps — dots.ocr parse with fitz fallback,
chunking, VLM describe, multimodal vectorization, Milvus+PG+asset storage,
report). The CLI keeps its argparse/dry-run/guard semantics unchanged; this
module only ADDS two optional parameters:

- ``collection``: target Milvus collection name (None = config default) for
  the dynamic-collection v1 (§8.5.3). Recorded in PG metadata so deletes and
  the /documents listing can route back to the right collection.
- ``extra_metadata``: extra keys merged into the ``rag_documents.metadata``
  JSONB (upload marks ``status``/``collection`` there; facets in
  ``GET /documents/filters`` aggregate straight from metadata, so they
  reflect the new document immediately after the PG write).

Size/page limits come from config (``MULTIMODAL_UPLOAD_MAX_MB`` /
``MULTIMODAL_UPLOAD_MAX_PAGES``, defaults identical to the former hardcoded
200MB / 500 pages) so the CLI and the upload endpoint enforce one policy.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from core.config import config
from loguru import logger


class UploadRejectedError(ValueError):
    """Client-rejectable upload problem from the §16/§20.2 guards: oversize PDF,
    page cap, unguarded overwrite (non-PDF is caught at the API boundary).

    Subclasses ``ValueError`` so the CLI's ``except ValueError`` contract is
    unchanged; the upload API keys its 422-vs-502 decision on this TYPE instead
    of message substrings (review P2-1). Dim/collection/embedding store errors
    stay plain ``ValueError``/``RuntimeError`` → 502.
    """


def natural_sort_key(name: str) -> tuple:
    """ch2.pdf < ch10.pdf — digit runs compare as ints (no natsort dependency)."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name.lower()))


def build_book_meta(book: str | None, chapter_start: int, files: list[Path]) -> list[dict]:
    """Assign book_id / chapter_index / chapter_label per §6.1 (natural order)."""
    metas: list[dict] = []
    for offset, path in enumerate(files):
        stem = path.stem
        if book:
            book_id = book.strip()
            chapter_index = chapter_start + offset
        else:
            book_id = stem
            chapter_index = 1
        chapter_label = f"{book_id} · 第{chapter_index}章 · {stem}"
        metas.append(
            {
                "book_id": book_id,
                "chapter_index": chapter_index,
                "chapter_label": chapter_label,
            }
        )
    return metas


def _prev_next_context(chunks: list, index: int) -> tuple[str, str]:
    """Adjacent text for the describe step: nearest text chunks around position."""
    prev_text = ""
    for j in range(index - 1, -1, -1):
        if chunks[j].kind == "text" and chunks[j].text.strip():
            prev_text = chunks[j].text
            break
    next_text = ""
    for j in range(index + 1, len(chunks)):
        if chunks[j].kind == "text" and chunks[j].text.strip():
            next_text = chunks[j].text
            break
    return prev_text, next_text


def _pdf_page_count(pdf_path: Path) -> int:
    """Read the page count with fitz WITHOUT rasterizing (review P1-4 pre-check)."""
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        return int(doc.page_count)


async def _run_guards(
    pdf_path: Path,
    document_id: int,
    *,
    dry_run: bool,
    replace: bool,
    max_bytes: int,
    max_pages: int,
) -> list:
    """Guards (§16): size cap, unguarded-overwrite guard, page-cap pre-check, rasterize.

    Returns the rendered page images; raises UploadRejectedError on any
    client-rejectable violation. Pure move of the former guards block.
    """
    from tools.dots_ocr_client import load_page_images
    from tools.node_repository import get_pool

    pdf_size = pdf_path.stat().st_size
    if pdf_size > max_bytes:
        raise UploadRejectedError(f"PDF {pdf_path.name} is {pdf_size} bytes (> {max_bytes}); rejected")

    # Existing-document guard (§20.2): same id + source=multimodal_pdf requires --replace.
    if not dry_run:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT metadata FROM rag_documents WHERE id = $1", document_id
            )
        if row is None:
            meta = None
        else:
            # asyncpg returns JSONB as str unless a codec is registered; the
            # upload path ALWAYS finds the placeholder row here (allocated
            # moments ago), so parse defensively (deployment bug #1 on a
            # fresh database).
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta) if meta.strip() else {}
        if row and (meta or {}).get("source") == "multimodal_pdf" and not replace:
            raise UploadRejectedError(
                f"document_id {document_id} already holds a multimodal document; rerun with --replace"
            )

    # Page-count pre-check BEFORE rasterizing (review P1-4): load_page_images
    # renders EVERY page into memory, so a huge PDF must be rejected from the
    # fitz page_count alone instead of OOM-ing in the renderer first.
    try:
        page_count = await asyncio.to_thread(_pdf_page_count, pdf_path)
    except Exception as exc:
        # A corrupt / fake .pdf is a client error, not a store failure (422,
        # not 502 — deployment finding #2).
        raise UploadRejectedError(
            f"cannot open {pdf_path.name} as a PDF: {type(exc).__name__}"
        ) from exc
    if page_count > max_pages:
        raise UploadRejectedError(f"PDF {pdf_path.name} has {page_count} pages (> {max_pages}); rejected")

    page_images, _sizes = await asyncio.to_thread(load_page_images, pdf_path, config.dot_ocr_dpi)
    if len(page_images) > max_pages:  # invariant: the fitz pre-check above should reject first
        raise UploadRejectedError(f"PDF {pdf_path.name} has {len(page_images)} pages (> {max_pages}); rejected")
    return page_images


async def _describe_and_store_assets(
    document_id: int,
    chunks: list,
    pages: list,
    summary: dict,
) -> dict[int, dict[str, bytes]]:
    """Steps 3+5a: save page/crop assets and VLM-describe image chunks (pure move).

    Mutates image chunks in place (text/metadata/image_ref) and
    ``summary["failed_images"]``; returns the page → crop-name → bytes map
    needed again by the vectorize step.
    """
    import base64

    from tools.multimodal_asset_store import get_asset_store
    from tools.multimodal_vlm import describe_image

    assets = get_asset_store()

    # Full page images are first-class assets (§6 layout): evidence thumbnails
    # and the fitz-fallback path's only visual product. Saved for BOTH parse
    # backends; keys {document_id}/page_{n}.jpg are backend-independent.
    for page in pages:
        if page.page_image_jpg:
            assets.save(document_id, f"page_{page.page_no}.jpg", page.page_image_jpg)

    # Crop names come from layout_to_md placeholders; map name → crop bytes per page.
    crops_by_page: dict[int, dict[str, bytes]] = {p.page_no: dict(p.picture_crops) for p in pages}
    failed_images = 0
    for index, chunk in enumerate(chunks):
        if chunk.kind != "image":
            continue
        crop = crops_by_page.get(chunk.page_no, {}).get(chunk.image_name or "")
        if crop is None:
            failed_images += 1
            summary["failed_images"] = failed_images
            chunk.text = f"（插图缺失：{chunk.image_name}）"
            continue
        image_ref = assets.save(document_id, chunk.image_name or "", crop)
        chunk.image_ref = image_ref
        prev_text, next_text = _prev_next_context(chunks, index)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(crop).decode("ascii")
        description, mode_used = await asyncio.to_thread(describe_image, data_uri, prev_text, next_text)
        chunk.text = description or f"（插图：{chunk.image_name}）"
        chunk.metadata["describe_mode"] = mode_used
    return crops_by_page


async def _vectorize_and_upsert(
    document_id: int,
    chunks: list,
    crops_by_page: dict[int, dict[str, bytes]],
    *,
    pdf_path: Path,
    book_meta: dict,
    target_collection: str,
) -> dict[str, Any]:
    """Steps 4+5: build rows, vectorize and upsert to Milvus (pure move)."""
    import base64

    from tools.multimodal_vectorizer import build_vectorizer
    from tools.retrieval_backends.dense_milvus_multimodal import (
        MilvusMultimodalDenseBackend,
        MmChunkRow,
    )

    rows: list[MmChunkRow] = []
    for c in chunks:
        data_uri = None
        if c.kind == "image":
            crop = crops_by_page.get(c.page_no, {}).get(c.image_name or "")
            if crop is None:
                continue  # missing crop already counted in failed_images
            data_uri = "data:image/jpeg;base64," + base64.b64encode(crop).decode("ascii")
        rows.append(
            MmChunkRow(
                kind=c.kind,
                page_no=c.page_no,
                title=c.title,
                text=c.text,
                category=c.category,
                image_ref=c.image_ref,
                image_data_uri=data_uri,
            )
        )
    backend = MilvusMultimodalDenseBackend(collection=target_collection)

    def _upsert_in_worker_thread() -> dict[str, Any]:
        """Run the vectorize-then-upsert on a NESTED event loop in a worker thread.

        Verified before choosing this shape (review P1-1): upsert_document_nodes
        and its call chain touch ONLY Milvus (sync pymilvus ensure/replace/
        insert) plus the HTTP vectorizer — never the asyncpg pool — so a second
        event loop here is safe. The vectorizer gets its own rate limiter
        because the module-level shared limiter's asyncio.Lock is bound to the
        caller's loop and would raise "bound to a different event loop" under
        contention (per-upload 120 RPM during ingest instead of a shared
        budget — acceptable for the offline ingest path).
        """
        from tools.multimodal_vectorizer import FixedWindowRateLimiter

        async def _run() -> dict[str, Any]:
            vectorizer = build_vectorizer(limiter=FixedWindowRateLimiter(config.multimodal_embed_rpm))
            try:
                return await backend.upsert_document_nodes(
                    document_id,
                    rows,
                    vectorizer=vectorizer,
                    filename=pdf_path.name,
                    book_id=book_meta["book_id"],
                    chapter_label=book_meta["chapter_label"],
                )
            finally:
                await vectorizer.aclose()

        return asyncio.run(_run())

    return await asyncio.to_thread(_upsert_in_worker_thread)


async def ingest_one_pdf(
    pdf_path: Path,
    document_id: int,
    book_meta: dict,
    *,
    dry_run: bool = False,
    replace: bool = False,
    collection: str | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Run the six-step pipeline for one PDF; returns the summary row.

    ``collection`` routes the Milvus writes (default = config.multimodal_collection);
    ``extra_metadata`` is merged into the rag_documents.metadata JSONB.
    UploadRejectedError = client-rejectable problem (oversize PDF, page cap,
    unguarded overwrite); other exceptions = upstream/store failures.

    Sync blocking segments (fitz rendering, dots.ocr HTTP probes, the parse
    ThreadPoolExecutor, the describe VLM call and the pymilvus upsert) run in
    worker threads via ``asyncio.to_thread`` so the upload API's event loop is
    never blocked (review P1-1). The function stays async; callers unchanged.
    """
    from tools.dots_ocr_client import DotsOcrClient
    from tools.node_repository import upsert_document

    target_collection = (collection or config.multimodal_collection or "").strip()
    max_bytes = max(1, int(config.multimodal_upload_max_mb)) * 1024 * 1024
    max_pages = max(1, int(config.multimodal_upload_max_pages))

    t0 = time.perf_counter()
    summary: dict = {
        "document_id": document_id,
        "filename": pdf_path.name,
        "pages": 0,
        "chunks_text": 0,
        "chunks_image": 0,
        "filtered_pages": 0,
        "failed_images": 0,
        "elapsed_s": 0.0,
        "backend": None,
        "collection": target_collection,
        "dry_run": dry_run,
    }

    # ── guards (§16) ──────────────────────────────────────────────────
    page_images = await _run_guards(
        pdf_path,
        document_id,
        dry_run=dry_run,
        replace=replace,
        max_bytes=max_bytes,
        max_pages=max_pages,
    )
    summary["pages"] = len(page_images)

    # ── step 1: parse ────────────────────────────────────────────────
    client = DotsOcrClient()
    if await asyncio.to_thread(client.healthy):
        pages = await asyncio.to_thread(client.parse_pdf, pdf_path, page_images)
        summary["backend"] = "dots_ocr"
    elif config.dot_ocr_fallback_fitz:
        pages = await asyncio.to_thread(client.parse_pdf_fitz_fallback, pdf_path, page_images)
        summary["backend"] = "fitz_fallback"
    else:
        raise RuntimeError(f"dots.ocr vLLM unreachable at {client.base_url!r} and fallback disabled")
    summary["filtered_pages"] = sum(1 for p in pages if p.filtered)

    # ── step 2: chunk ────────────────────────────────────────────────
    from tools.multimodal_chunker import chunk_document

    if dry_run:
        chunks = await chunk_document(pages, embed_fn=None)
    else:
        # Semantic-split embedder: the ARK multimodal endpoint with text-only
        # content blocks (deployment finding #7). The old path went through the
        # TEXT embedding provider — a SECOND vendor dependency that broke the
        # server deployment (zhipu has no balance; openrouter unreachable) even
        # though chunk vectorization itself is ark-only. Using the same
        # multimodal space as retrieval also makes the split signal consistent
        # with what search actually ranks on. Own rate limiter: this runs on
        # the caller's loop where the shared limiter's lock is safe, but the
        # limiter instance should not outlive the chunking step.
        from tools.multimodal_vectorizer import build_vectorizer

        chunk_vectorizer = build_vectorizer()
        try:

            async def _embed_for_chunking(texts: list[str]) -> list[list[float]]:
                results = await chunk_vectorizer.embed_texts(texts)
                return [r.vector for r in results if r.vector is not None]

            chunks = await chunk_document(pages, embed_fn=_embed_for_chunking)
        finally:
            await chunk_vectorizer.aclose()
    summary["chunks_text"] = sum(1 for c in chunks if c.kind == "text")
    summary["chunks_image"] = sum(1 for c in chunks if c.kind == "image")

    if dry_run:
        summary["elapsed_s"] = round(time.perf_counter() - t0, 2)
        summary["estimated_calls"] = {
            "parse_pages": len(pages),
            "vlm_describe": summary["chunks_image"],
            "embedding": summary["chunks_text"] + summary["chunks_image"],
        }
        return summary

    # ── steps 3+5a: describe images & save assets ────────────────────
    crops_by_page = await _describe_and_store_assets(document_id, chunks, pages, summary)

    # ── steps 4+5: vectorize & store ─────────────────────────────────
    upsert_stats = await _vectorize_and_upsert(
        document_id,
        chunks,
        crops_by_page,
        pdf_path=pdf_path,
        book_meta=book_meta,
        target_collection=target_collection,
    )
    summary["upsert"] = upsert_stats

    # ── step 5b: PG registration (rag_documents metadata hierarchy §6.1) ──
    metadata: dict[str, Any] = {
        "source": "multimodal_pdf",
        "pages": summary["pages"],
        "chunks": {"text": summary["chunks_text"], "image": summary["chunks_image"]},
        "dpi": config.dot_ocr_dpi,
        "embedding_model": config.multimodal_embedding_model,
        "book_id": book_meta["book_id"],
        "chapter_index": book_meta["chapter_index"],
        "chapter_label": book_meta["chapter_label"],
        "parse_backend": summary["backend"],
        "collection": target_collection,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    try:
        await upsert_document(
            document_id,
            title=book_meta["chapter_label"],
            source_uri=str(pdf_path),
            file_type="multimodal_pdf",
            metadata=metadata,
        )
    except Exception:
        # PG registration failed AFTER the Milvus points were written: best-effort
        # compensating rollback so a retry starts from a clean store (review
        # stretch item). Cleanup failures are logged, never mask the PG error.
        from tools.multimodal_asset_store import get_asset_store
        from tools.retrieval_backends.dense_milvus_multimodal import (
            MilvusMultimodalDenseBackend,
        )

        logger.exception("[MmIngest] PG registration failed for {}; rolling back stores", document_id)
        try:
            await asyncio.to_thread(
                MilvusMultimodalDenseBackend(collection=target_collection).replace_document_nodes,
                document_id,
            )
        except Exception as cleanup_exc:
            logger.warning("[MmIngest] Milvus rollback failed for {}: {}", document_id, cleanup_exc)
        try:
            await asyncio.to_thread(get_asset_store().delete_document, document_id)
        except Exception as cleanup_exc:
            logger.warning("[MmIngest] asset rollback failed for {}: {}", document_id, cleanup_exc)
        raise

    # ── step 6: report ───────────────────────────────────────────────
    from tools.multimodal_asset_store import get_asset_store

    report_dir = Path(config.multimodal_pages_dir) / str(document_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "document_id": document_id,
        "filename": pdf_path.name,
        "backend": summary["backend"],
        "pages": summary["pages"],
        "filtered_pages": summary["filtered_pages"],
        "chunks_text": summary["chunks_text"],
        "chunks_image": summary["chunks_image"],
        "failed_images": summary.get("failed_images", 0),
        "filtered_page_errors": [p.error for p in pages if p.filtered],
        "book": book_meta,
        "collection": target_collection,
        "asset_bytes": get_asset_store().document_bytes(document_id),
    }
    (report_dir / "ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["elapsed_s"] = round(time.perf_counter() - t0, 2)
    logger.info("[MmIngest] document {} stored into collection {}", document_id, target_collection)
    return summary
