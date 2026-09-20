"""Factories for retrieval backend selection.

M5 reality: dense = ``milvus`` | ``milvus_multimodal``; sparse = ``milvus`` |
``postgres`` | ``none``. Qdrant and OpenSearch were removed (fail-fast errors
below name the removal for stale envs).
"""

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
    if backend == "milvus_multimodal":
        # Multimodal dense has no shared sparse rows: the combo matrix is
        # fail-fast (design §4.2) — none is the only legal sparse partner.
        sparse = (config.sparse_backend or "").strip().lower()
        if sparse in ("milvus", "postgres"):
            raise ValueError(
                f"DENSE_BACKEND=milvus_multimodal requires SPARSE_BACKEND=none; got {sparse!r} "
                "(multimodal dense writes no rag_nodes sparse rows)"
            )
        from .dense_milvus_multimodal import MilvusMultimodalDenseBackend

        return MilvusMultimodalDenseBackend()
    raise ValueError(
        f"Unsupported dense backend: {config.dense_backend!r} "
        "(qdrant was removed at M5; dense retrieval is milvus|milvus_multimodal)"
    )


@lru_cache(maxsize=1)
def get_sparse_backend() -> SparseBackend:
    backend = (config.sparse_backend or "milvus").strip().lower()
    if backend == "milvus":
        return MilvusSparseBackend()
    if backend == "postgres":
        return PostgresSparseBackend()
    if backend == "none":
        from .sparse_none import NoneSparseBackend

        return NoneSparseBackend()
    raise ValueError(
        f"Unsupported sparse backend: {config.sparse_backend!r} "
        "(opensearch was removed at M5-prime; sparse retrieval is milvus|postgres|none)"
    )
