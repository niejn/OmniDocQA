#!/usr/bin/env python3
"""Asset GC — reconcile local asset dirs against Milvus ``rag_multimodal`` image refs.

Design §11 (phase-2 batch, runnable standalone): the delete cascade (§20.1)
can leave orphaned asset files when a step fails mid-cascade; this script is
the reconciliation backstop. Orphans = files on disk that no Milvus point
references anymore (by document_id dir / image_ref basename).

Two-phase by default:
  dry-run (default) → prints the orphan list + byte totals, deletes nothing
  --apply           → deletes the reported orphans (after your review)

Data-safety interlocks (--apply aborts unless ``--force``):
  1. Milvus collection missing  → the reconciliation is empty by construction;
     every disk asset would be judged an orphan and the whole asset dir wiped.
  2. Empty reconciliation + non-empty disk → almost certainly a wrong
     collection/env, not a real "everything is orphaned" situation.
  3. Truncated reconciliation → the Milvus scan is capped at
     ``_MILVUS_QUERY_LIMIT`` rows; when the collection reports more rows than
     were actually fetched, the orphan list is incomplete (files referenced by
     unfetched rows would be deleted as orphans).

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
from typing import NamedTuple

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

# Milvus query cap — must match the backend's own query limits; going beyond
# needs the truncation check below, not a bigger blind limit.
_MILVUS_QUERY_LIMIT = 16384


class Reconciliation(NamedTuple):
    """Milvus-side view of referenced assets + how complete the scan was."""

    referenced: dict[int, set[str]]  # document_id → referenced basenames
    doc_ids: set[int]  # document ids that still have points
    collection_exists: bool
    milvus_row_count: int | None  # collection stats row count; None when unknown
    fetched_rows: int  # rows actually returned by the reconciliation query


def referenced_assets() -> Reconciliation:
    """Scan Milvus for referenced image assets (see :class:`Reconciliation`)."""
    from tools.retrieval_backends.dense_milvus_multimodal import (
        MilvusMultimodalDenseBackend,
    )

    MilvusMultimodalDenseBackend()
    client = get_client()
    if not client.has_collection(config.multimodal_collection):
        return Reconciliation({}, set(), False, None, 0)
    rows = client.query(
        collection_name=config.multimodal_collection,
        filter="document_id > 0",
        output_fields=["document_id", "image_ref"],
        limit=_MILVUS_QUERY_LIMIT,
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
    try:
        stats = client.get_collection_stats(collection_name=config.multimodal_collection)
        milvus_row_count = int((stats or {}).get("row_count") or 0)
    except Exception as exc:
        logger.warning("[AssetGC] collection stats unavailable ({}): truncation check skipped", exc)
        milvus_row_count = None
    return Reconciliation(referenced, doc_ids, True, milvus_row_count, len(rows))


def on_disk_assets() -> dict[int, list[Path]]:
    root = Path(config.multimodal_pages_dir)
    if not root.is_dir():
        return {}
    out: dict[int, list[Path]] = {}
    for doc_dir in root.iterdir():
        if doc_dir.is_dir() and doc_dir.name.isdigit():
            out[int(doc_dir.name)] = sorted(f for f in doc_dir.iterdir() if f.is_file())
    return out


def _safety_interlocks(
    *,
    apply_mode: bool,
    recon: Reconciliation,
    disk: dict[int, list[Path]],
) -> tuple[list[str], list[str]]:
    """Return (warnings, aborts): warnings print in every mode, aborts fire --apply.

    Each interlock guards against a reconciliation that is empty or partial
    while the disk still holds assets — deleting "orphans" in that state would
    destroy live data.
    """
    warnings: list[str] = []
    aborts: list[str] = []
    if not recon.collection_exists:
        warnings.append(
            f"MILVUS COLLECTION MISSING: {config.multimodal_collection!r} does not exist — "
            "the reconciliation is empty by construction and every disk asset would be judged an orphan."
        )
        if apply_mode:
            aborts.append("collection missing: refusing to --apply against an empty reconciliation (use --force to override)")
    else:
        if not recon.referenced and not recon.doc_ids and disk:
            warnings.append(
                "EMPTY RECONCILIATION: Milvus reports no documents but the disk holds "
                f"{len(disk)} document dir(s) — likely a wrong collection/env, not real orphans."
            )
            if apply_mode:
                aborts.append("empty reconciliation with non-empty disk: refusing to --apply (use --force to override)")
        if recon.milvus_row_count is None:
            warnings.append("row count unavailable: query-truncation check skipped (completeness unverified)")
        elif recon.fetched_rows != recon.milvus_row_count:
            warnings.append(
                f"TRUNCATED RECONCILIATION: fetched {recon.fetched_rows} rows but the collection reports "
                f"{recon.milvus_row_count} — the {_MILVUS_QUERY_LIMIT}-row query cap cut the scan; "
                "the orphan list is incomplete (live assets may be misjudged as orphans)."
            )
            if apply_mode:
                aborts.append("reconciliation truncated by query limit: refusing to --apply (use --force to override)")
    return warnings, aborts


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile multimodal assets vs Milvus (GC)")
    parser.add_argument("--apply", action="store_true", help="Actually delete orphans (default dry-run)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="DANGEROUS: bypass the --apply safety aborts (missing/empty/truncated Milvus "
        "reconciliation). Only after manual review of a dry-run on the SAME env.",
    )
    args = parser.parse_args()

    recon = referenced_assets()
    disk = on_disk_assets()

    orphans: list[Path] = []
    missing: list[str] = []

    # Orphans: files under doc dirs that Milvus no longer references,
    # or whole dirs whose document has no points left.
    for doc_id, files in disk.items():
        refs = recon.referenced.get(doc_id, set())
        if doc_id not in recon.doc_ids and not refs:
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
    for doc_id, names in recon.referenced.items():
        for name in sorted(names):
            if not (Path(config.multimodal_pages_dir) / str(doc_id) / name).is_file():
                missing.append(f"{doc_id}/{name}")

    orphan_bytes = sum(f.stat().st_size for f in orphans)
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "docs_on_disk": len(disk),
        "docs_in_milvus": len(recon.doc_ids),
        "milvus_rows_reported": recon.milvus_row_count,
        "milvus_rows_fetched": recon.fetched_rows,
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

    # Safety interlocks run after the report so operators see what was (not)
    # reconciled, but always BEFORE any deletion. --force suppresses the
    # --apply aborts while the warnings still print.
    warnings, aborts = _safety_interlocks(
        apply_mode=args.apply and not args.force, recon=recon, disk=disk
    )
    for w in warnings:
        logger.warning("[AssetGC] {}", w)
        print(f"WARNING: {w}")
    for reason in aborts:
        logger.error("[AssetGC] {}", reason)
        print(f"ABORT: {reason}")
    if aborts:
        print(json.dumps({"aborted": True, "reasons": aborts}))
        raise SystemExit(2)

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
