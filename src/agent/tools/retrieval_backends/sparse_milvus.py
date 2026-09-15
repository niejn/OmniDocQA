"""Milvus-backed sparse retrieval backend (Step 1B: BM25 over the shared text field).

The ``rag_nodes`` collection rows are written by ``MilvusDenseBackend``'s
replace path — every insert carries ``text`` and Milvus's BM25 function
derives the sparse vector server-side — so ``replace_document_nodes`` here is
a no-op. Searches send the raw query text over the ``sparse`` field; Milvus
applies the collection analyzer (``MILVUS_TEXT_ANALYZER``) and scores BM25.

Requires ``DENSE_BACKEND=milvus`` (enforced in the factory): the schema's
dense vector column is non-nullable, so no path writes Milvus rows without
the dense backend.
"""

from __future__ import annotations

import asyncio

from ..milvus_store import sparse_search
from .sparse_query_profiles import SparseQueryPlan
from .types import MetadataFilters, NodeHit, NodeIndexRecord


class MilvusSparseBackend:
    async def search(
        self,
        document_ids: list[int],
        query: str,
        *,
        limit: int,
        levels: list[int] | None = None,
        parent_ids: list[str] | None = None,
        metadata_filters: MetadataFilters | None = None,
        query_plan: SparseQueryPlan | None = None,
        log_stage: str | None = None,
    ) -> list[NodeHit]:
        return await asyncio.to_thread(
            sparse_search,
            query,
            document_ids=document_ids,
            limit=limit,
            levels=levels,
            parent_ids=parent_ids,
            metadata_filters=metadata_filters,
            query_plan=query_plan,
            log_stage=log_stage,
        )

    async def replace_document_nodes(self, document_id: int, nodes: list[NodeIndexRecord]) -> None:
        # Rows (id/text/dense + BM25 sparse) are maintained by the Milvus
        # dense backend's replace path; nothing to write here.
        return None
