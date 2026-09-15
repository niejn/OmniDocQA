"""Factories for retrieval backend selection."""

from __future__ import annotations

from functools import lru_cache

from core.config import config

from .dense_milvus import MilvusDenseBackend
from .sparse_milvus import MilvusSparseBackend
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
    backend = (config.sparse_backend or "milvus").strip().lower()
    if backend == "milvus":
        return MilvusSparseBackend()
    if backend == "postgres":
        return PostgresSparseBackend()
    raise ValueError(
        f"Unsupported sparse backend: {config.sparse_backend!r} "
        "(opensearch was removed at M5-prime; sparse retrieval is milvus|postgres)"
    )
