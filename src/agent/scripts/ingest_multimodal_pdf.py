#!/usr/bin/env python3
"""Multimodal PDF ingest CLI (T1.5) — the six-step pipeline orchestrator.

PDF → parse (dots.ocr vLLM / fitz fallback) → chunk (headers/pictures/semantic)
→ describe images (VLM layered) → vectorize (multimodal embedding) → store
(Milvus rag_multimodal + PG rag_documents + local assets) → observability.

CLI contract (design §16):
  --data-dir --glob "*.pdf" --document-id-start N [--book NAME] [--chapter-start N]
  [--no-fallback] [--dry-run] [--replace]
Exit codes: 0 ok (filtered pages included) | 2 parse unreachable & no fallback
| 3 vectorize/store failure | 4 param/file error.
Idempotent: same document_id rerun = delete points then full rewrite.
Guard: an existing multimodal document_id requires --replace (§20.2).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

MAX_PAGES = 500
MAX_PDF_BYTES = 200 * 1024 * 1024

EXIT_OK = 0
EXIT_PARSE_UNREACHABLE = 2
EXIT_STORE_FAILURE = 3
EXIT_PARAM_ERROR = 4


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


async def _embed_for_chunking(texts: list[str]) -> list[list[float]]:
    """Semantic-split embedder: reuse the TEXT embedding pipeline (chunk signal only)."""
    from tools.vectorizer import generate_embeddings_batch

    out = await generate_embeddings_batch(texts)
    return [v for v in out if v]


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


async def ingest_one_pdf(
    pdf_path: Path,
    document_id: int,
    book_meta: dict,
    *,
    dry_run: bool = False,
    replace: bool = False,
) -> dict:
    """Run the six-step pipeline for one PDF; returns the summary row."""
    from core.config import config
    from tools.dots_ocr_client import DotsOcrClient, load_page_images
    from tools.multimodal_asset_store import get_asset_store
    from tools.node_repository import get_pool, upsert_document

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
        "dry_run": dry_run,
    }

    # ── guards (§16) ──────────────────────────────────────────────────
    pdf_size = pdf_path.stat().st_size
    if pdf_size > MAX_PDF_BYTES:
        raise ValueError(f"PDF {pdf_path.name} is {pdf_size} bytes (> {MAX_PDF_BYTES}); rejected")

    # Existing-document guard (§20.2): same id + source=multimodal_pdf requires --replace.
    if not dry_run:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT metadata FROM rag_documents WHERE id = $1", document_id
            )
        if row and (row["metadata"] or {}).get("source") == "multimodal_pdf" and not replace:
            raise ValueError(
                f"document_id {document_id} already holds a multimodal document; rerun with --replace"
            )

    page_images, sizes = load_page_images(pdf_path, config.dot_ocr_dpi)
    summary["pages"] = len(page_images)
    if len(page_images) > MAX_PAGES:
        raise ValueError(f"PDF {pdf_path.name} has {len(page_images)} pages (> {MAX_PAGES}); rejected")

    # ── step 1: parse ────────────────────────────────────────────────
    client = DotsOcrClient()
    if client.healthy():
        pages = client.parse_pdf(pdf_path, page_images=page_images)
        summary["backend"] = "dots_ocr"
    elif config.dot_ocr_fallback_fitz:
        pages = client.parse_pdf_fitz_fallback(pdf_path, page_images=page_images)
        summary["backend"] = "fitz_fallback"
    else:
        raise RuntimeError(f"dots.ocr vLLM unreachable at {client.base_url!r} and fallback disabled")
    summary["filtered_pages"] = sum(1 for p in pages if p.filtered)

    # ── step 2: chunk ────────────────────────────────────────────────
    from tools.multimodal_chunker import chunk_document

    if dry_run:
        chunks = await chunk_document(pages, embed_fn=None)
    else:
        chunks = await chunk_document(pages, embed_fn=_embed_for_chunking)
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
    import base64

    assets = get_asset_store()
    from tools.multimodal_vlm import describe_image

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
        description, mode_used = describe_image(data_uri, prev_text, next_text)
        chunk.text = description or f"（插图：{chunk.image_name}）"
        chunk.metadata["describe_mode"] = mode_used

    # ── steps 4+5: vectorize & store ─────────────────────────────────
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
    vectorizer = build_vectorizer()
    backend = MilvusMultimodalDenseBackend()
    try:
        upsert_stats = await backend.upsert_document_nodes(
            document_id,
            rows,
            vectorizer=vectorizer,
            filename=pdf_path.name,
            book_id=book_meta["book_id"],
            chapter_label=book_meta["chapter_label"],
        )
    except Exception:
        await vectorizer.aclose()
        raise
    await vectorizer.aclose()
    summary["upsert"] = upsert_stats

    # ── step 5b: PG registration (rag_documents metadata hierarchy §6.1) ──
    await upsert_document(
        document_id,
        title=book_meta["chapter_label"],
        source_uri=str(pdf_path),
        file_type="multimodal_pdf",
        metadata={
            "source": "multimodal_pdf",
            "pages": summary["pages"],
            "chunks": {"text": summary["chunks_text"], "image": summary["chunks_image"]},
            "dpi": config.dot_ocr_dpi,
            "embedding_model": config.multimodal_embedding_model,
            "book_id": book_meta["book_id"],
            "chapter_index": book_meta["chapter_index"],
            "chapter_label": book_meta["chapter_label"],
            "parse_backend": summary["backend"],
        },
    )

    # ── step 6: report ───────────────────────────────────────────────
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
        "failed_images": failed_images,
        "filtered_page_errors": [p.error for p in pages if p.filtered],
        "book": book_meta,
        "asset_bytes": assets.document_bytes(document_id),
    }
    (report_dir / "ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return summary


async def run(args: argparse.Namespace) -> int:
    from loguru import logger
    from tools.rag_stage_log import log_rag

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(json.dumps({"error": f"data dir not found: {data_dir}"}))
        return EXIT_PARAM_ERROR
    files = sorted(data_dir.glob(args.glob), key=lambda p: natural_sort_key(p.name))
    if not files:
        print(json.dumps({"error": f"no files match {args.glob!r} under {data_dir}"}))
        return EXIT_PARAM_ERROR
    if args.book is None and len(files) > 1:
        print(
            json.dumps(
                {
                    "error": "multiple files matched without --book: they become separate books; pass --book to aggregate chapters",
                }
            )
        )
        return EXIT_PARAM_ERROR

    book_metas = build_book_meta(args.book, args.chapter_start, files)
    results: list[dict] = []
    exit_code = EXIT_OK
    for pdf_path, book_meta in zip(files, book_metas):
        document_id = args.document_id_start + (
            book_meta["chapter_index"] - args.chapter_start if args.book else 0
        )
        try:
            summary = await ingest_one_pdf(
                pdf_path,
                document_id,
                book_meta,
                dry_run=args.dry_run,
                replace=args.replace,
            )
            results.append(summary)
            log_rag("ingest_mm_done", **{k: v for k, v in summary.items() if k != "upsert"})
            print(json.dumps({"ok": True, **summary}, ensure_ascii=False))
        except RuntimeError as exc:  # parse unreachable & no fallback
            logger.error("[IngestMm] parse unreachable: {}", exc)
            print(json.dumps({"ok": False, "document_id": document_id, "error": str(exc)}, ensure_ascii=False))
            exit_code = EXIT_PARSE_UNREACHABLE
            break
        except ValueError as exc:
            if "embedding" in str(exc) or "collection" in str(exc) or "dim" in str(exc):
                logger.error("[IngestMm] store failure: {}", exc)
                print(json.dumps({"ok": False, "document_id": document_id, "error": str(exc)}, ensure_ascii=False))
                exit_code = EXIT_STORE_FAILURE
                break
            logger.error("[IngestMm] rejected: {}", exc)
            print(json.dumps({"ok": False, "document_id": document_id, "error": str(exc)}, ensure_ascii=False))
            exit_code = EXIT_PARAM_ERROR
            break
        except Exception as exc:
            logger.exception("[IngestMm] store failure")
            print(json.dumps({"ok": False, "document_id": document_id, "error": str(exc)[:500]}, ensure_ascii=False))
            exit_code = EXIT_STORE_FAILURE
            break
    print(json.dumps({"done": True, "count": len(results), "results": results}, ensure_ascii=False))
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the multimodal document library")
    parser.add_argument("--data-dir", required=True, help="Directory containing source PDFs")
    parser.add_argument("--glob", default="*.pdf", help="Filename glob (default: *.pdf)")
    parser.add_argument("--document-id-start", type=int, required=True, help="First document_id")
    parser.add_argument("--book", default=None, help="Aggregate matched files as chapters of one book")
    parser.add_argument("--chapter-start", type=int, default=1, help="First chapter number (default 1)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable fitz degradation (parse must be up)")
    parser.add_argument("--dry-run", action="store_true", help="Parse + chunk only; no model calls, cost estimate")
    parser.add_argument("--replace", action="store_true", help="Allow overwriting an existing multimodal document_id")
    args = parser.parse_args()

    if args.no_fallback:
        from core.config import config

        config.dot_ocr_fallback_fitz = False

    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
