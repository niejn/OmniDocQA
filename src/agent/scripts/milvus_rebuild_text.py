#!/usr/bin/env python3
"""Rewrite the Milvus BM25 `text` column under the current enrichment config.

Reads every row back (dense vector included), strips the old enrichment
prefix, rebuilds text = title×N + search_hints×M + body, and upserts.
Idempotent: re-running with unchanged config rewrites nothing.

Usage (from src/agent):
    python scripts/milvus_rebuild_text.py --dry-run          # preview change counts
    python scripts/milvus_rebuild_text.py                    # rewrite all documents
    python scripts/milvus_rebuild_text.py --document-ids 9833,9801
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from core.config import config  # noqa: E402
from tools.milvus_store import get_client, rebuild_texts  # noqa: E402


def _parse_document_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return sorted({int(part) for part in raw.split(",") if part.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Milvus BM25 text enrichment in place")
    parser.add_argument("--dry-run", action="store_true", help="count changes without upserting")
    parser.add_argument("--document-ids", type=str, default=None, help="comma-separated subset, e.g. 9833,9801")
    args = parser.parse_args()

    stats = rebuild_texts(document_ids=_parse_document_ids(args.document_ids), dry_run=args.dry_run)
    stats.pop("changed_by_doc", None)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    client = get_client()
    row_count = client.get_collection_stats(config.milvus_collection).get("row_count")
    print(f"\ncollection row_count after: {row_count}")


if __name__ == "__main__":
    main()
