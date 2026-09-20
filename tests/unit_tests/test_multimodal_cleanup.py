"""P1-3 regression: the DELETE cascade must clear vector stores and assets BEFORE the PG row.

The old order deleted the PG row first, so a later-step failure left the
document 404-forever (multimodal_document_exists gates the retry) and pushed
the retry down the SEC branch. Everything runs against fakes — no PG, no
Milvus, no MinIO.
"""

from __future__ import annotations

import asyncio

import pytest
from core.config import config
from tools import multimodal_cleanup
from tools.multimodal_cleanup import DocumentCascadeError, delete_document_everywhere

ORDER: list[str] = []


class FakeConn:
    source: str | None = "multimodal_pdf"
    pg_fails = False

    async def fetchrow(self, query: str, *args: object):
        return {"source": FakeConn.source, "collection": "mm_col"}

    async def execute(self, query: str, *args: object):
        if FakeConn.pg_fails:
            raise RuntimeError("pg gone")
        ORDER.append("pg")
        return "DELETE 1"


class FakePool:
    def acquire(self):
        class _Ctx:
            async def __aenter__(self):
                return FakeConn()

            async def __aexit__(self, *exc: object) -> bool:
                return False

        return _Ctx()


class FakeMmBackend:
    def __init__(self, collection: str | None = None) -> None:
        self.collection = collection

    def replace_document_nodes(self, document_id: int, *args: object) -> None:
        ORDER.append("milvus")


class FakeDenseBackend:
    def replace_document_nodes(self, document_id: int, nodes=None) -> None:
        ORDER.append("dense")


class FakeSparseBackend:
    async def replace_document_nodes(self, document_id: int, nodes=None) -> None:
        ORDER.append("sparse")


class FakeAssetStore:
    def document_bytes(self, document_id: int) -> int:
        return 123

    def delete_document(self, document_id: int) -> None:
        ORDER.append("assets")


@pytest.fixture(autouse=True)
def _reset():
    ORDER.clear()
    FakeConn.source = "multimodal_pdf"
    FakeConn.pg_fails = False
    yield
    ORDER.clear()


@pytest.fixture()
def patched(monkeypatch: pytest.MonkeyPatch):
    import tools.multimodal_asset_store as asset_store
    import tools.node_repository as node_repo
    import tools.retrieval_backends.dense_milvus as dense_mod
    import tools.retrieval_backends.dense_milvus_multimodal as mm_mod
    import tools.retrieval_backends.sparse_milvus as sparse_mod

    async def fake_get_pool():
        return FakePool()

    monkeypatch.setattr(node_repo, "get_pool", fake_get_pool)
    monkeypatch.setattr(mm_mod, "MilvusMultimodalDenseBackend", FakeMmBackend)
    monkeypatch.setattr(asset_store, "get_asset_store", lambda: FakeAssetStore())
    monkeypatch.setattr(dense_mod, "MilvusDenseBackend", FakeDenseBackend)
    monkeypatch.setattr(sparse_mod, "MilvusSparseBackend", FakeSparseBackend)
    monkeypatch.setattr(config, "sparse_backend", "milvus")
    return monkeypatch


def test_multimodal_cascade_deletes_pg_last(patched) -> None:
    result = asyncio.run(delete_document_everywhere(9801))

    # Vector points → assets → PG row LAST (retryable partial failures).
    assert ORDER == ["milvus", "assets", "pg"]
    assert result["deleted_pg_rows"] == 1
    assert result["multimodal_deleted"] is True
    assert result["assets_deleted"] is True
    assert result["collection"] == "mm_col"


def test_text_cascade_deletes_pg_last(patched) -> None:
    FakeConn.source = None  # no multimodal marker → SEC text branch

    result = asyncio.run(delete_document_everywhere(9801))

    assert ORDER == ["dense", "sparse", "pg"]
    assert result == {
        "document_id": 9801,
        "deleted_pg_rows": 1,
        "dense_deleted": True,
        "sparse_deleted": True,
    }


def test_skip_pg_keeps_vector_and_asset_cleanup(patched) -> None:
    result = asyncio.run(delete_document_everywhere(9801, skip_pg=True))

    assert ORDER == ["milvus", "assets"]  # read-only detect + stores still cleaned
    assert result["deleted_pg_rows"] is None
    assert result["multimodal_deleted"] is True


def test_vector_failure_leaves_pg_row_for_retry(patched) -> None:
    class BoomMm(FakeMmBackend):
        def replace_document_nodes(self, document_id: int, *args: object) -> None:
            raise RuntimeError("milvus down")

    import tools.retrieval_backends.dense_milvus_multimodal as mm_mod

    patched.setattr(mm_mod, "MilvusMultimodalDenseBackend", BoomMm)

    with pytest.raises(DocumentCascadeError) as excinfo:
        asyncio.run(delete_document_everywhere(9801))
    assert excinfo.value.step == multimodal_cleanup.STEP_MILVUS
    assert "pg" not in ORDER  # PG row survives → the DELETE can be retried


def test_pg_failure_still_reports_postgres_step_after_stores_cleaned(patched) -> None:
    FakeConn.pg_fails = True

    with pytest.raises(DocumentCascadeError) as excinfo:
        asyncio.run(delete_document_everywhere(9801))
    assert excinfo.value.step == multimodal_cleanup.STEP_POSTGRES
    # Vector + asset cleanup already succeeded before the PG write failed.
    assert ORDER == ["milvus", "assets"]
