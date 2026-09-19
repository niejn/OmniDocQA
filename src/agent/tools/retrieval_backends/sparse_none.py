"""Null sparse backend for DENSE_BACKEND=milvus_multimodal (T2.2).

The multimodal collection has no PG sparse rows to query — sparse search is
semantically empty on that path. Returning an explicit no-op backend keeps the
factory matrix total (milvus|postgres|none) without conditional None handling
at every call site. v1 queries the multimodal collection dense-only anyway.
"""

from __future__ import annotations

from typing import Any

from .types import NodeIndexRecord, SparseQueryPlan


class NoneSparseBackend:
    async def search(
        self,
        document_ids: list[int],
        query: str,
        *,
        limit: int,
        levels: list[int] | None = None,
        parent_ids: list[str] | None = None,
        metadata_filters: dict[str, list[str]] | None = None,
        query_plan: SparseQueryPlan | None = None,
        log_stage: str | None = None,
    ) -> list[dict[str, Any]]:
        return []

    async def replace_document_nodes(self, document_id: int, nodes: list[NodeIndexRecord]) -> None:
        return None
