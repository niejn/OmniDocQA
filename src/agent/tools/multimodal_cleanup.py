"""Document deletion cascade (§20.1 / §8.5.2) — shared by CLI and DELETE API.

Extracted from ``scripts/delete_ingested_document.py`` so the CLI and
``DELETE /agent/api/documents/documents/{id}`` run the identical cascade:

- multimodal documents: Milvus points (of the collection recorded in PG
  metadata, default config.multimodal_collection) → asset dir/objects → PG row.
- SEC text documents: dense (shared Milvus rows) → sparse backend → PG row.

PG goes LAST in both cascades (review P1-3): deleting the PG row first made a
later step failure unrecoverable — the document 404'd out of
``multimodal_document_exists`` so the cascade could never be retried, and the
next attempt took the SEC branch for a multimodal document. With PG last, a
partial failure leaves the PG row in place and the retry re-runs the
idempotent vector/asset steps.

Each step is idempotent. Failures raise :class:`DocumentCascadeError` with a
``step`` attribute so the API can report WHICH step failed (§8.5.2: 502 with
best-effort cleanup detail); the CLI surfaces it as any other exception.
Sync Milvus/asset calls are pushed to a worker thread so the API's event loop
is never blocked (review fix A1).
"""

from __future__ import annotations

import asyncio

from core.config import config
from loguru import logger

STEP_DETECT = "detect"
STEP_POSTGRES = "postgres"
STEP_MILVUS = "milvus"
STEP_ASSETS = "assets"
STEP_DENSE = "dense"
STEP_SPARSE = "sparse"


class DocumentCascadeError(RuntimeError):
    """Cascade step failure; ``step`` names the failing stage for the 502 detail."""

    def __init__(self, message: str, *, step: str) -> None:
        super().__init__(message)
        self.step = step


async def _detect(document_id: int) -> tuple[str | None, str | None]:
    """Read (source, collection) from rag_documents.metadata — read-only, so it
    also runs under --skip-pg (skipping it would miss multimodal points/assets)."""
    from tools.node_repository import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT metadata->>'source' AS source, metadata->>'collection' AS collection "
            "FROM rag_documents WHERE id = $1",
            document_id,
        )
    if row is None:
        return None, None
    return row["source"], row["collection"]


async def _delete_pg_row(document_id: int) -> int:
    """Final cascade step: DELETE the rag_documents row; returns the row count."""
    from tools.node_repository import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            cmd = await conn.execute("DELETE FROM rag_documents WHERE id = $1", document_id)
        except Exception as exc:
            raise DocumentCascadeError(f"PG delete failed: {exc}", step=STEP_POSTGRES) from exc
    return int(cmd.split()[-1])


async def delete_document_everywhere(
    document_id: int,
    *,
    skip_pg: bool = False,
    collection: str | None = None,
) -> dict:
    """Delete one document across PG + vector backends + assets; returns the result row.

    ``collection`` overrides the Milvus multimodal collection recorded in PG
    metadata (the API passes it when the caller knows the target collection).
    Order: vector stores → assets → PG row LAST (retryable partial failures,
    review P1-3); ``skip_pg`` only skips that final PG DELETE write.
    """
    source, meta_collection = await _detect(document_id)
    is_multimodal = source == "multimodal_pdf"

    if is_multimodal:
        # Multimodal cascade (§20.1): Milvus points → assets → PG row. Each step
        # idempotent; the sparse/dense rag_nodes paths below don't apply (no
        # rag_nodes rows exist for multimodal documents).
        from tools.multimodal_asset_store import get_asset_store
        from tools.retrieval_backends.dense_milvus_multimodal import (
            MilvusMultimodalDenseBackend,
        )

        target = (collection or meta_collection or config.multimodal_collection or "").strip()
        try:
            await asyncio.to_thread(
                MilvusMultimodalDenseBackend(collection=target).replace_document_nodes, document_id
            )
        except Exception as exc:
            raise DocumentCascadeError(
                f"Milvus delete failed on collection {target!r}: {exc}", step=STEP_MILVUS
            ) from exc

        def _purge_assets() -> int:
            # delete_document is an idempotent no-op when the dir/objects are gone;
            # report whether the store actually held bytes for this document
            # instead of hard-coding True.
            store = get_asset_store()
            present = store.document_bytes(document_id) > 0
            store.delete_document(document_id)
            return 1 if present else 0

        try:
            assets_present = await asyncio.to_thread(_purge_assets)
        except Exception as exc:
            raise DocumentCascadeError(f"asset cleanup failed: {exc}", step=STEP_ASSETS) from exc

        deleted_pg = None if skip_pg else await _delete_pg_row(document_id)
        return {
            "document_id": document_id,
            "deleted_pg_rows": deleted_pg,
            "multimodal_deleted": True,
            "collection": target,
            "assets_deleted": bool(assets_present),
            "assets_absent": not assets_present,
        }

    # Dense backend (milvus-only since M5; deletion removes the shared rows)
    from tools.retrieval_backends.dense_milvus import MilvusDenseBackend

    try:
        await asyncio.to_thread(MilvusDenseBackend().replace_document_nodes, document_id, [])
    except Exception as exc:
        raise DocumentCascadeError(f"dense delete failed: {exc}", step=STEP_DENSE) from exc

    # Sparse backend (milvus rows are shared with dense; postgres keeps nodes in PG)
    sparse_backend_name = (config.sparse_backend or "milvus").strip().lower()
    try:
        if sparse_backend_name == "milvus":
            from tools.retrieval_backends.sparse_milvus import MilvusSparseBackend

            # No-op: dense=milvus delete above already removed the shared rows.
            await MilvusSparseBackend().replace_document_nodes(document_id, [])
        elif sparse_backend_name == "postgres":
            from tools.retrieval_backends.sparse_postgres import PostgresSparseBackend

            await PostgresSparseBackend().replace_document_nodes(document_id, [])
        elif sparse_backend_name == "none":
            pass  # multimodal mode: nothing to clean on the sparse path
        else:
            raise ValueError(f"Unsupported sparse backend: {config.sparse_backend!r}")
    except DocumentCascadeError:
        raise
    except Exception as exc:
        raise DocumentCascadeError(f"sparse delete failed: {exc}", step=STEP_SPARSE) from exc

    deleted_pg = None if skip_pg else await _delete_pg_row(document_id)
    logger.info("[DocumentCleanup] deleted document {} (dense+sparse paths)", document_id)
    return {
        "document_id": document_id,
        "deleted_pg_rows": deleted_pg,
        "dense_deleted": True,
        "sparse_deleted": True,
    }
