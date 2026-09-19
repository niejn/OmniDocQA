#!/usr/bin/env python3
"""Asset GC — reconcile local asset dirs against Milvus ``rag_multimodal`` image refs.

Design §11 (phase-2 batch, runnable standalone): the delete cascade (§20.1)
can leave orphaned asset files when a step fails mid-cascade; this script is
the reconciliation backstop. Orphans = files on disk that no Milvus point
references anymore (by document_id dir / image_ref basename).

Two-phase by default:
  dry-run (default) → prints the orphan list + byte totals, deletes nothing
  --apply           → deletes the reported orphans (after your review)

Also flags referenced-but-missing assets (Milvus image_ref without a file) —
those degrade to text-only evidence cards in the UI and are listed for
re-ingest decisions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))


from core.config import config  # noqa: E402
from loguru import logger  # noqa: E402
from tools.milvus_store import get_client  # noqa: E402

# Non-asset metadata files that live inside the asset dir but are never
# referenced from Milvus (pipeline bookkeeping, not images).
_KEEP_FILES = {"ingest_report.json"}
# page_{n}.jpg assets are CONVENTION-addressed (document_id + page_no from
# hits), not image_ref-addressed: legitimate while the document still has
# points, orphans only when the whole document is gone.
_PAGE_FILE = re.compile(r"^page_\d+\.jpg$")


def referenced_assets() -> tuple[dict[int, set[str]], set[int]]:
    """image_ref (document_id, basename) → from Milvus; plus all doc ids with points."""
    from tools.retrieval_backends.dense_milvus_multimodal import (
        MilvusMultimodalDenseBackend,
    )

    MilvusMultimodalDenseBackend()
    client = get_client()
    if not client.has_collection(config.multimodal_collection):
        return {}, set()
    rows = client.query(
        collection_name=config.multimodal_collection,
        filter="document_id > 0",
        output_fields=["document_id", "image_ref"],
        limit=16384,
    )
    referenced: dict[int, set[str]] = defaultdict(set)
    doc_ids: set[int] = set()
    for row in rows:
        doc_id = int(row["document_id"])
        doc_ids.add(doc_id)
        image_ref = str(row.get("image_ref") or "")
        if image_ref:
            _, _, name = image_ref.partition("/")
            referenced[doc_id].add(name)
    return referenced, doc_ids


def on_disk_assets() -> dict[int, list[Path]]:
    root = Path(config.multimodal_pages_dir)
    if not root.is_dir():
        return {}
    out: dict[int, list[Path]] = {}
    for doc_dir in root.iterdir():
        if doc_dir.is_dir() and doc_dir.name.isdigit():
            out[int(doc_dir.name)] = sorted(f for f in doc_dir.iterdir() if f.is_file())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile multimodal assets vs Milvus (GC)")
    parser.add_argument("--apply", action="store_true", help="Actually delete orphans (default dry-run)")
    args = parser.parse_args()

    referenced, doc_ids = referenced_assets()
    disk = on_disk_assets()

    orphans: list[Path] = []
    missing: list[str] = []

    # Orphans: files under doc dirs that Milvus no longer references,
    # or whole dirs whose document has no points left.
    for doc_id, files in disk.items():
        refs = referenced.get(doc_id, set())
        if doc_id not in doc_ids and not refs:
            orphans.extend(f for f in files if f.name not in _KEEP_FILES)  # whole document gone
            continue
        for f in files:
            if f.name in _KEEP_FILES:
                continue
            if _PAGE_FILE.match(f.name):
                continue  # convention-addressed; doc still has points
            if f.name not in refs:
                orphans.append(f)
    # Referenced-but-missing: image_ref with no file (evidence cards degrade).
    for doc_id, names in referenced.items():
        for name in sorted(names):
            if not (Path(config.multimodal_pages_dir) / str(doc_id) / name).is_file():
                missing.append(f"{doc_id}/{name}")

    orphan_bytes = sum(f.stat().st_size for f in orphans)
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "docs_on_disk": len(disk),
        "docs_in_milvus": len(doc_ids),
        "orphans": [str(f) for f in orphans],
        "orphan_count": len(orphans),
        "orphan_bytes": orphan_bytes,
        "referenced_missing": missing,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "orphans"}, ensure_ascii=False))
    for f in orphans:
        print(f"  orphan: {f}")
    for m in missing:
        print(f"  referenced-but-missing: {m}")

    if orphans and args.apply:
        for f in orphans:
            f.unlink(missing_ok=True)
        # remove now-empty doc dirs
        for doc_id in list(disk):
            doc_dir = Path(config.multimodal_pages_dir) / str(doc_id)
            if doc_dir.is_dir() and not any(doc_dir.iterdir()):
                doc_dir.rmdir()
        logger.info("[AssetGC] deleted {} orphans ({} bytes)", len(orphans), orphan_bytes)
        print(json.dumps({"deleted": len(orphans), "bytes": orphan_bytes}))
    elif orphans:
        print(f"dry-run: rerun with --apply to delete {len(orphans)} orphans ({orphan_bytes} bytes)")


if __name__ == "__main__":
    main()
