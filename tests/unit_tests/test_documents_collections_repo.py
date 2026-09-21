"""Unit tests for the document_collections registry + inventory mappers (§8.5).

All PG access runs against a fake pool (no server); the collection-name
validator and the overview→item mapper are pure functions.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import asyncpg
import pytest
from tools.documents import document_repository as repo
from tools.documents.document_repository import DocumentSetError

# ── fake pool / conn (asyncpg-shaped, just enough for these queries) ─────


class FakeConn:
    """Routes the handful of SQL shapes the repository module issues."""

    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}
        self.executed: list[str] = []
        self.max_document_id = 0
        self.documents: dict[int, bool] = {}

    async def execute(self, query: str, *args: object) -> str:
        self.executed.append(" ".join(query.split()))
        head = query.strip().upper()
        if head.startswith("CREATE TABLE"):
            return "CREATE TABLE"
        if head.startswith("DELETE"):
            return "DELETE 1"
        return "OK"

    async def fetchrow(self, query: str, *args: object):
        sql = " ".join(query.split())
        if "MAX(id)" in sql:
            return {"max_id": self.max_document_id}
        if "RETURNING" in sql:  # INSERT ... RETURNING for the registry
            name = str(args[0])
            if name in self.collections:
                raise asyncpg.UniqueViolationError(
                    "duplicate key value violates unique constraint"
                )
            row = {
                "collection_name": name,
                "embedding_provider": args[1],
                "description": args[2],
                "created_at": datetime(2026, 9, 20, tzinfo=UTC),
            }
            self.collections[name] = row
            return dict(row)
        if "WHERE collection_name = $1" in sql:
            found = self.collections.get(str(args[0]))
            return dict(found) if found else None
        if "FROM rag_documents WHERE id = $1" in sql:
            return {"1": 1} if self.documents.get(int(args[0])) else None
        raise AssertionError(f"unexpected query in FakeConn: {sql}")

    async def fetch(self, query: str, *args: object):
        if "FROM document_collections" in query:
            return [dict(r) for r in self.collections.values()]
        raise AssertionError(f"unexpected query in FakeConn: {query}")


class FakePool:
    def __init__(self) -> None:
        self.conn = FakeConn()

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self) -> FakeConn:
                return pool.conn

            async def __aexit__(self, *exc: object) -> bool:
                return False

        return _Ctx()


@pytest.fixture()
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> FakePool:
    pool = FakePool()

    async def _get_pool() -> FakePool:
        return pool

    monkeypatch.setattr(repo, "get_pool", _get_pool)
    return pool


# ── collections registry CRUD ────────────────────────────────────────────


def test_ensure_collections_table_is_idempotent_create(fake_pool: FakePool) -> None:
    asyncio.run(repo.ensure_collections_table())
    asyncio.run(repo.ensure_collections_table())
    creates = [q for q in fake_pool.conn.executed if q.startswith("CREATE TABLE")]
    assert len(creates) == 2
    assert all("document_collections" in q for q in creates)


def test_insert_and_get_dynamic_collection(fake_pool: FakePool) -> None:
    row = asyncio.run(
        repo.insert_dynamic_collection(
            collection_name="flink_course", embedding_provider="ark", description=None
        )
    )
    assert row["collection_name"] == "flink_course"
    assert row["created_at"] == "2026-09-20T00:00:00+00:00"
    got = asyncio.run(repo.get_dynamic_collection("flink_course"))
    assert got is not None and got["collection_name"] == "flink_course"
    assert asyncio.run(repo.get_dynamic_collection("missing")) is None


def test_insert_duplicate_raises_unique_violation(fake_pool: FakePool) -> None:
    async def _twice() -> None:
        await repo.insert_dynamic_collection(collection_name="dup", embedding_provider=None, description=None)
        await repo.insert_dynamic_collection(collection_name="dup", embedding_provider=None, description=None)

    with pytest.raises(asyncpg.UniqueViolationError):
        asyncio.run(_twice())


def test_list_dynamic_collections(fake_pool: FakePool) -> None:
    asyncio.run(
        repo.insert_dynamic_collection(collection_name="a_col", embedding_provider=None, description=None)
    )
    asyncio.run(
        repo.insert_dynamic_collection(collection_name="b_col", embedding_provider="ark", description="d")
    )
    names = [r["collection_name"] for r in asyncio.run(repo.list_dynamic_collections())]
    assert sorted(names) == ["a_col", "b_col"]


# ── inventory: exists + overview mapper ──────────────────────────────────


def test_multimodal_document_exists(fake_pool: FakePool) -> None:
    fake_pool.conn.documents = {42: True, 43: False}
    assert asyncio.run(repo.multimodal_document_exists(42)) is True
    assert asyncio.run(repo.multimodal_document_exists(43)) is False


def test_validate_collection_name_matrix() -> None:
    for good in ("abc", "flink_course", "a1_b2", "x" * 32):
        assert repo.validate_collection_name(good) == good
    for bad in ("", "Abc", "1abc", "ab", "x" * 33, "bad-name", "bad name", "has.dot"):
        with pytest.raises(DocumentSetError):
            repo.validate_collection_name(bad)


def test_build_document_list_item_maps_metadata() -> None:
    row = {
        "document_id": 9801,
        "title": "Flink · 第1章 · ch1",
        "source_uri": "C:/books/ch1.pdf",
        "metadata": {
            "source": "multimodal_pdf",
            "pages": 14,
            "chunks": {"text": 30, "image": 6},
            "book_id": "Apache Flink",
            "chapter_label": "Flink · 第1章 · ch1",
        },
        "created_at": "2026-09-20T01:02:03+00:00",
    }
    item = repo.build_document_list_item(row)
    assert item == {
        "document_id": 9801,
        "status": "completed",
        "filename": "ch1.pdf",
        "title": "Flink · 第1章 · ch1",
        "page_count": 14,
        "node_count": 36,
        "book_id": "Apache Flink",
        "chapter_label": "Flink · 第1章 · ch1",
        "created_at": "2026-09-20T01:02:03+00:00",
    }


def test_build_document_list_item_fallbacks() -> None:
    item = repo.build_document_list_item(
        {
            "document_id": 7,
            "title": None,
            "source_uri": None,
            "metadata": {"source": "multimodal_pdf"},
            "created_at": "2026-09-20T00:00:00+00:00",
        }
    )
    assert item["page_count"] == 0
    assert item["node_count"] == 0
    assert item["book_id"] == "7"  # falls back to the document id
    assert item["filename"] == ""
