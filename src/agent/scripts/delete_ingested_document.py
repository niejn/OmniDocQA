#!/usr/bin/env python3
"""Delete ingested document data from PG + vector backends.

Thin CLI shell over ``tools.multimodal_cleanup.delete_document_everywhere``
(§8.5.2: the DELETE API shares the same cascade). CLI contract unchanged:
--document-id (one or more) [--skip-pg] — --skip-pg only skips the PG DELETE
write; the read-only metadata lookup still runs so multimodal documents are
detected and their Milvus points + asset dirs are cleared.
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

from tools.multimodal_cleanup import delete_document_everywhere  # noqa: E402


async def main_async(document_ids: list[int], *, skip_pg: bool) -> dict:
    from tools.node_repository import ensure_schema

    await ensure_schema()
    out = []
    for did in document_ids:
        out.append(await delete_document_everywhere(did, skip_pg=skip_pg))
    return {"success": True, "results": out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete ingested document data by document_id")
    parser.add_argument(
        "--document-id",
        type=int,
        nargs="+",
        required=True,
        help="One or more document IDs to delete",
    )
    parser.add_argument(
        "--skip-pg",
        action="store_true",
        help="Skip the rag_documents DELETE write (a read-only metadata lookup still runs so "
        "multimodal documents are detected); only clear vector/asset backends",
    )
    args = parser.parse_args()

    result = asyncio.run(main_async(args.document_id, skip_pg=args.skip_pg))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
