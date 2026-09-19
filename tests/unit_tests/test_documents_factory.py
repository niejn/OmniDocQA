"""Unit tests for the backend matrix (T2.2): milvus_multimodal combos, none sparse, ask guard."""

import pytest
from core.config import config
from tools.retrieval_backends import factory
from tools.retrieval_backends.dense_milvus_multimodal import (
    MilvusMultimodalDenseBackend,
)
from tools.retrieval_backends.sparse_none import NoneSparseBackend


@pytest.fixture(autouse=True)
def reset_factory_cache():
    factory.get_dense_backend.cache_clear()
    factory.get_sparse_backend.cache_clear()
    yield
    factory.get_dense_backend.cache_clear()
    factory.get_sparse_backend.cache_clear()


def test_default_milvus_milvus_unchanged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "dense_backend", "milvus")
    monkeypatch.setattr(config, "sparse_backend", "milvus")
    from tools.retrieval_backends.dense_milvus import MilvusDenseBackend
    from tools.retrieval_backends.sparse_milvus import MilvusSparseBackend

    assert isinstance(factory.get_dense_backend(), MilvusDenseBackend)
    assert isinstance(factory.get_sparse_backend(), MilvusSparseBackend)


def test_multimodal_with_none_is_legal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "dense_backend", "milvus_multimodal")
    monkeypatch.setattr(config, "sparse_backend", "none")
    assert isinstance(factory.get_dense_backend(), MilvusMultimodalDenseBackend)
    assert isinstance(factory.get_sparse_backend(), NoneSparseBackend)


@pytest.mark.parametrize("sparse", ["milvus", "postgres"])
def test_multimodal_with_sparse_rows_fails_fast(monkeypatch: pytest.MonkeyPatch, sparse: str):
    monkeypatch.setattr(config, "dense_backend", "milvus_multimodal")
    monkeypatch.setattr(config, "sparse_backend", sparse)
    with pytest.raises(ValueError, match="SPARSE_BACKEND=none"):
        factory.get_dense_backend()


def test_none_sparse_alone_with_text_dense_is_legal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "dense_backend", "milvus")
    monkeypatch.setattr(config, "sparse_backend", "none")
    assert isinstance(factory.get_sparse_backend(), NoneSparseBackend)


def test_unknown_dense_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "dense_backend", "qdrant")
    with pytest.raises(ValueError, match="Unsupported dense backend"):
        factory.get_dense_backend()


def test_ask_guard_blocks_multimodal_mode(monkeypatch: pytest.MonkeyPatch):
    """§10-2: /ask raises a clear ValueError under DENSE_BACKEND=milvus_multimodal."""
    from tools.rag_service import answer_question

    monkeypatch.setattr(config, "dense_backend", "milvus_multimodal")
    with pytest.raises(ValueError, match="/agent/api/documents"):
        __import__("asyncio").run(
            answer_question(question="q", document_ids=[1], detail_level="brief", top_k=3)
        )
