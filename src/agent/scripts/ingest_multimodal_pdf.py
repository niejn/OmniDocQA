#!/usr/bin/env python3
"""Multimodal PDF ingest CLI (T1.5) — thin shell over ``tools.multimodal_ingest``.

The six-step pipeline (parse → chunk → describe → vectorize → store → report)
lives in ``tools/multimodal_ingest.py`` so the upload API (§8.5.2) shares it.
This CLI keeps its contract unchanged (design §16):
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
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from tools.multimodal_ingest import (  # noqa: E402  (re-exported for tests/CLI parity)
    UploadRejectedError,
    build_book_meta,
    ingest_one_pdf,
    natural_sort_key,
)

EXIT_OK = 0
EXIT_PARSE_UNREACHABLE = 2
EXIT_STORE_FAILURE = 3
EXIT_PARAM_ERROR = 4


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
        except UploadRejectedError as exc:  # §16/§20.2 guards: param-class rejection
            logger.error("[IngestMm] rejected: {}", exc)
            print(json.dumps({"ok": False, "document_id": document_id, "error": str(exc)}, ensure_ascii=False))
            exit_code = EXIT_PARAM_ERROR
            break
        except ValueError as exc:  # store failure: dim/collection/embedding family (§16)
            logger.error("[IngestMm] store failure: {}", exc)
            print(json.dumps({"ok": False, "document_id": document_id, "error": str(exc)}, ensure_ascii=False))
            exit_code = EXIT_STORE_FAILURE
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
