"""Milvus-backed dense retrieval implementation (Qdrant parity)."""

from __future__ import annotations

from ..milvus_store import delete_document_nodes, dense_search, insert_nodes
from .types import MetadataFilters, NodeHit, NodeIndexRecord


class MilvusDenseBackend:
    def search(
        self,
        query_vector: list[float],
        *,
        document_ids: list[int],
        limit: int,
        levels: list[int] | None = None,
        parent_ids: list[str] | None = None,
        metadata_filters: MetadataFilters | None = None,
        log_stage: str | None = None,
    ) -> list[NodeHit]:
        return dense_search(
            query_vector,
            document_ids=document_ids,
            limit=limit,
            levels=levels,
            parent_ids=parent_ids,
            metadata_filters=metadata_filters,
            log_stage=log_stage,
        )

    def replace_document_nodes(self, document_id: int, nodes: list[NodeIndexRecord]) -> None:
        delete_document_nodes(document_id)
        if nodes:
            insert_nodes(nodes)
