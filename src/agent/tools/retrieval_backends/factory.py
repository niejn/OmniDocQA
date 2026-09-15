"""Factories for retrieval backend selection."""

from __future__ import annotations

from functools import lru_cache

from core.config import config

from .dense_milvus import MilvusDenseBackend
from .sparse_milvus import MilvusSparseBackend
from .sparse_opensearch import OpenSearchSparseBackend
from .sparse_postgres import PostgresSparseBackend
from .types import DenseBackend, SparseBackend


@lru_cache(maxsize=1)
def get_dense_backend() -> DenseBackend:
    backend = (config.dense_backend or "milvus").strip().lower()
    if backend == "milvus":
        return MilvusDenseBackend()
    raise ValueError(
        f"Unsupported dense backend: {config.dense_backend!r} "
        "(qdrant was removed at M5; dense retrieval is Milvus-only)"
    )


@lru_cache(maxsize=1)
def get_sparse_backend() -> SparseBackend:
    backend = (config.sparse_backend or "postgres").strip().lower()
    if backend == "milvus":
        dense = (config.dense_backend or "milvus").strip().lower()
        if dense != "milvus":
            raise ValueError(
                "SPARSE_BACKEND=milvus requires DENSE_BACKEND=milvus: the shared "
                "rag_nodes rows (text + BM25 sparse vector) are only written by "
                "the Milvus dense backend."
            )
        return MilvusSparseBackend()
    if backend == "postgres":
        return PostgresSparseBackend()
    if backend == "opensearch":
        return OpenSearchSparseBackend()
    raise ValueError(f"Unsupported sparse backend: {config.sparse_backend!r}")
