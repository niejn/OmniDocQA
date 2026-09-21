"""API contract tests for /agent/api/documents (§8.5.2 upload + §8.5.3/§8.5.4).

Everything runs offline through FastAPI TestClient with faked repositories,
faked Milvus backends and a faked ingest/testset core — no PG, no Milvus,
no model calls.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from core.config import config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tools.documents import document_api, document_service
from tools.documents.document_service import DocumentAskError
from tools.multimodal_cleanup import DocumentCascadeError

# ── fakes ────────────────────────────────────────────────────────────────


class FakeRepo:
    """The slice of document_repository the service layer touches."""

    def __init__(self) -> None:
        self.dynamic: list[dict] = []
        self.overview: list[dict] = []
        self.exists = True
        self.docs: list[dict] = []
        self.sets: list[dict] = []
        self.placeholders: list[dict] = []
        self.status_marks: list[dict] = []
        self.deleted_placeholders: list[int] = []
        self.fail_next_placeholder = False

    async def get_dynamic_collection(self, name: str) -> dict | None:
        return next((d for d in self.dynamic if d["collection_name"] == name), None)

    async def list_dynamic_collections(self) -> list[dict]:
        return list(self.dynamic)

    async def insert_dynamic_collection(
        self, *, collection_name: str, embedding_provider: str | None, description: str | None
    ) -> dict:
        row = {
            "collection_name": collection_name,
            "embedding_provider": embedding_provider,
            "description": description,
            "created_at": "2026-09-20T00:00:00+00:00",
        }
        self.dynamic.append(row)
        return row

    async def multimodal_document_exists(self, document_id: int) -> bool:
        return self.exists

    async def multimodal_document_overview(self) -> list[dict]:
        return list(self.overview)

    async def list_multimodal_documents(self) -> list[dict]:
        return self.docs

    async def get_set(self, set_id: str) -> dict | None:
        return None

    async def expand_books_to_doc_ids(self, book_ids):
        return [], list(book_ids)

    async def expand_chapters(self, document_ids):
        return [], list(document_ids)

    async def insert_placeholder_document(
        self, document_id: int, *, collection: str, filename: str, book_meta: dict | None = None
    ) -> None:
        if self.fail_next_placeholder:
            self.fail_next_placeholder = False
            import asyncpg

            raise asyncpg.UniqueViolationError("rag_documents_pkey")
        self.placeholders.append(
            {
                "document_id": int(document_id),
                "collection": collection,
                "filename": filename,
                "book_meta": book_meta,
            }
        )

    async def delete_placeholder_document(self, document_id: int) -> bool:
        self.deleted_placeholders.append(int(document_id))
        return True

    async def mark_document_ingest_status(self, document_id: int, status: str) -> bool:
        self.status_marks.append({"document_id": int(document_id), "status": status})
        return True


class FakeAssetStore:
    """Stands in for get_asset_store so the API tests stay offline even when
    .env points MULTIMODAL_ASSET_STORE at an unreachable MinIO."""

    def open(self, document_id: int, name: str) -> bytes:
        raise KeyError(f"{document_id}/{name}")

    def document_bytes(self, document_id: int) -> int:
        return 0

    def delete_document(self, document_id: int) -> None:
        return None


class FakeVectorizer:
    async def embed_text(self, text: str):
        class Embed:
            vector = [0.1, 0.2, 0.3]
            error = None

        return Embed()

    async def aclose(self) -> None:
        return None


class FakeMmBackend:
    """Stands in for MilvusMultimodalDenseBackend; records constructor targets."""

    created: list[str] = []
    fail_ensure = False
    exists = False
    points = 0

    def __init__(self, collection: str | None = None) -> None:
        self.collection = collection or config.multimodal_collection

    def ensure_collection(self, vector_size: int | None = None) -> None:
        FakeMmBackend.created.append(self.collection)
        if FakeMmBackend.fail_ensure:
            raise RuntimeError("milvus down")

    def has_collection(self) -> bool:
        return FakeMmBackend.exists

    def count(self) -> int:
        return FakeMmBackend.points

    def search(self, vector, **kwargs: Any) -> list[dict]:
        return [
            {
                "chunk_id": "c1",
                "score": 0.9,
                "kind": "text",
                "document_id": 9801,
                "filename": "ch1.pdf",
                "title": "t",
                "page_no": 0,
                "text_preview": "hello",
                "book_id": "b",
                "chapter_label": "l",
            }
        ]


class FakeMilvusClient:
    def has_collection(self, name: str) -> bool:
        return False


@pytest.fixture()
def fake_repo(monkeypatch: pytest.MonkeyPatch) -> FakeRepo:
    repo = FakeRepo()
    monkeypatch.setattr(document_service, "repo", repo)
    return repo


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, fake_repo: FakeRepo) -> Iterator[TestClient]:
    monkeypatch.setattr(document_service, "build_vectorizer", lambda: FakeVectorizer())
    monkeypatch.setattr(document_service, "MilvusMultimodalDenseBackend", FakeMmBackend)
    monkeypatch.setattr(document_service, "get_asset_store", lambda: FakeAssetStore())
    FakeMmBackend.created = []
    FakeMmBackend.fail_ensure = False

    import tools.milvus_store as milvus_store

    monkeypatch.setattr(milvus_store, "get_client", lambda: FakeMilvusClient())

    app = FastAPI()
    app.include_router(document_api.router)
    # Context-manager form keeps ONE portal/event loop alive across requests so
    # asyncio.create_task background jobs actually run between requests.
    with TestClient(app) as test_client:
        yield test_client


def _wait_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/documents/testset/{job_id}").json()
        if payload.get("status") in ("completed", "failed"):
            return payload
        time.sleep(0.02)
    raise AssertionError("job did not settle in time")


# ── upload (§8.5.2) ──────────────────────────────────────────────────────


def test_upload_happy_path_contract(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    async def fake_upload(*, pdf_path, collection):
        captured["collection"] = collection
        return {
            "document_id": 9901,
            "status": "completed",
            "node_count": 36,
            "page_count": 14,
            "filename": "ch1.pdf",
        }

    monkeypatch.setattr(document_api, "upload_multimodal_pdf", fake_upload)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("ch1.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "document_id": 9901,
        "status": "completed",
        "node_count": 36,
        "page_count": 14,
        "filename": "ch1.pdf",
    }
    # The endpoint passes the raw form value through; default resolution is the
    # service layer's job (covered by test_upload_service_defaults_and_metadata).
    assert captured["collection"] is None


def test_upload_rejects_wrong_field_name(client: TestClient) -> None:
    # Contract: multipart field MUST be `file`; anything else is a 422.
    response = client.post(
        "/api/documents/upload",
        files={"document": ("ch1.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 422


def test_upload_422_matrix(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. non-PDF filename
    response = client.post(
        "/api/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 422
    assert "message" in response.json()["detail"]

    # 2. oversize (> MULTIMODAL_UPLOAD_MAX_MB)
    monkeypatch.setattr(config, "multimodal_upload_max_mb", 1)
    big = b"x" * (1024 * 1024 + 1)
    response = client.post(
        "/api/documents/upload", files={"file": ("big.pdf", big, "application/pdf")}
    )
    assert response.status_code == 422

    # 3. unknown collection (fake repo holds no registry entries)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("ch1.pdf", b"%PDF-1.4", "application/pdf")},
        data={"collection": "no_such_collection"},
    )
    assert response.status_code == 422


def test_upload_store_failure_is_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*, pdf_path, collection):
        raise RuntimeError("dots.ocr unreachable and no embedding provider")

    monkeypatch.setattr(document_api, "upload_multimodal_pdf", boom)
    response = client.post(
        "/api/documents/upload", files={"file": ("ch1.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert response.status_code == 502
    assert "message" in response.json()["detail"]


def test_upload_cleans_temp_dir(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mkdtemp upload directory is removed after the request, not just the pdf (P2-3)."""
    captured: dict = {}

    async def fake_upload(*, pdf_path, collection):
        captured["pdf"] = Path(pdf_path)
        return {
            "document_id": 1,
            "status": "completed",
            "node_count": 1,
            "page_count": 1,
            "filename": "ch1.pdf",
        }

    monkeypatch.setattr(document_api, "upload_multimodal_pdf", fake_upload)
    response = client.post(
        "/api/documents/upload", files={"file": ("ch1.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert response.status_code == 200
    assert not captured["pdf"].exists()  # pdf gone ...
    assert not captured["pdf"].parent.exists()  # ... and the whole mkdtemp dir with it
    assert captured["pdf"].parent.name.startswith("mm_upload_")


def test_jobs_cap_evicts_oldest_finished_only() -> None:
    """At 50 jobs the oldest completed/failed entry is evicted; running jobs stay (P2-7)."""
    saved = dict(document_service.JOBS)
    try:
        document_service.JOBS.clear()
        for i in range(50):
            document_service.JOBS[f"j{i}"] = {
                "job_id": f"j{i}",
                "kind": "testset",
                # j0..j48 finished (oldest first), j49 still running
                "status": "completed" if i < 49 else "running",
                "created_at": f"2026-01-01T00:00:{i:02d}",
            }
        job_id = asyncio.run(document_service._register_job("evaluate"))
        assert len(document_service.JOBS) == 50  # one in, one evicted
        assert "j0" not in document_service.JOBS  # oldest finished evicted
        assert "j49" in document_service.JOBS  # running untouched
        assert document_service.JOBS[job_id]["status"] == "running"
    finally:
        document_service.JOBS.clear()
        document_service.JOBS.update(saved)


def test_classify_upload_failure_matrix() -> None:
    # 422 decided by EXCEPTION TYPE (review P2-1), not message substrings.
    from tools.multimodal_ingest import UploadRejectedError

    assert (
        document_service.classify_upload_failure(
            UploadRejectedError("PDF ch1.pdf has 900 pages (> 500); rejected")
        )
        == 422
    )
    assert (
        document_service.classify_upload_failure(
            UploadRejectedError("document_id 9806 already holds a multimodal document")
        )
        == 422
    )
    # dim/collection/embedding family stays plain ValueError → 502 store failure.
    assert document_service.classify_upload_failure(ValueError("dim=1536 != embedding dim=3072")) == 502
    assert document_service.classify_upload_failure(RuntimeError("milvus down")) == 502


def test_upload_service_defaults_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_repo: FakeRepo
) -> None:
    """Service layer: id allocation + placeholder reservation, default collection, extra_metadata markers."""
    import tools.multimodal_ingest as mm_ingest
    import tools.node_repository as node_repo

    class MaxIdConn:
        async def fetchrow(self, query: str, *args: object):
            assert "MAX(id)" in query
            return {"max_id": 9805}

    class MaxIdPool:
        def acquire(self):
            class _Ctx:
                async def __aenter__(self):
                    return MaxIdConn()

                async def __aexit__(self, *exc: object) -> bool:
                    return False

            return _Ctx()

    async def fake_pool() -> MaxIdPool:
        return MaxIdPool()

    captured: dict = {}

    async def fake_ingest(
        pdf_path,
        document_id,
        book_meta,
        *,
        dry_run=False,
        replace=False,
        collection=None,
        extra_metadata=None,
    ):
        captured.update(
            document_id=document_id,
            book_meta=book_meta,
            collection=collection,
            extra_metadata=extra_metadata,
            replace=replace,
        )
        return {
            "document_id": document_id,
            "filename": pdf_path.name,
            "pages": 12,
            "chunks_text": 30,
            "chunks_image": 6,
        }

    monkeypatch.setattr(node_repo, "get_pool", fake_pool)
    monkeypatch.setattr(mm_ingest, "ingest_one_pdf", fake_ingest)
    monkeypatch.setattr(document_service, "_milvus_document_exists", lambda doc_id, col: False)

    pdf = tmp_path / "第三章.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    result = asyncio.run(document_service.upload_multimodal_pdf(pdf_path=pdf, collection=None))

    assert result == {
        "document_id": 9806,  # MAX(id)+1
        "status": "completed",
        "node_count": 36,  # chunks_text + chunks_image
        "page_count": 12,
        "filename": "第三章.pdf",
    }
    # The id was RESERVED with a placeholder row (status=ingesting) before ingest;
    # the placeholder carries the book metadata so a failed upload stays
    # self-describing (facet shows the file, not a numeric-id ghost).
    assert fake_repo.placeholders == [
        {
            "document_id": 9806,
            "collection": config.multimodal_collection,
            "filename": "第三章.pdf",
            "book_meta": {
                "book_id": "第三章",
                "chapter_index": 1,
                "chapter_label": "第三章 · 第1章 · 第三章",
            },
        }
    ]
    assert captured["collection"] == config.multimodal_collection
    assert captured["book_meta"]["book_id"] == "第三章"
    # replace=True: the placeholder row is ours; a fresh id has no old points.
    assert captured["replace"] is True
    extra = captured["extra_metadata"]
    # source MUST match multimodal_cleanup._detect's marker exactly.
    assert extra["source"] == "multimodal_pdf"
    assert extra["collection"] == config.multimodal_collection
    assert extra["status"] == "completed"
    assert "uploaded_at" in extra
    # No failure marks on the happy path.
    assert fake_repo.status_marks == []


def test_allocate_document_id_retries_after_unique_violation(
    monkeypatch: pytest.MonkeyPatch, fake_repo: FakeRepo
) -> None:
    """The loser of the placeholder INSERT race bumps its candidate and retries (P1-2)."""
    import tools.node_repository as node_repo

    class MaxIdConn:
        async def fetchrow(self, query: str, *args: object):
            return {"max_id": 100}

    class MaxIdPool:
        def acquire(self):
            class _Ctx:
                async def __aenter__(self):
                    return MaxIdConn()

                async def __aexit__(self, *exc: object) -> bool:
                    return False

            return _Ctx()

    async def fake_pool():
        return MaxIdPool()

    monkeypatch.setattr(node_repo, "get_pool", fake_pool)
    monkeypatch.setattr(document_service, "_milvus_document_exists", lambda doc_id, col: False)
    fake_repo.fail_next_placeholder = True  # 101 collides, 102 wins

    allocated = asyncio.run(document_service.allocate_document_id("mm_col", filename="a.pdf"))

    assert allocated == 102
    assert [p["document_id"] for p in fake_repo.placeholders] == [102]


def test_upload_failure_flags_placeholder_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_repo: FakeRepo
) -> None:
    """Ingest failure UPDATEs the placeholder to status=failed and re-raises (P1-2)."""
    import tools.multimodal_ingest as mm_ingest
    import tools.node_repository as node_repo

    class MaxIdConn:
        async def fetchrow(self, query: str, *args: object):
            return {"max_id": 0}

    class MaxIdPool:
        def acquire(self):
            class _Ctx:
                async def __aenter__(self):
                    return MaxIdConn()

                async def __aexit__(self, *exc: object) -> bool:
                    return False

            return _Ctx()

    async def boom(*args: object, **kwargs: object):
        raise RuntimeError("dots.ocr unreachable")

    async def fake_pool():
        return MaxIdPool()

    monkeypatch.setattr(node_repo, "get_pool", fake_pool)
    monkeypatch.setattr(mm_ingest, "ingest_one_pdf", boom)
    monkeypatch.setattr(document_service, "_milvus_document_exists", lambda doc_id, col: False)

    pdf = tmp_path / "ch1.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(RuntimeError):
        asyncio.run(document_service.upload_multimodal_pdf(pdf_path=pdf, collection=None))

    assert fake_repo.placeholders[0]["document_id"] == 1  # row kept (DELETE cascade cleans it)
    assert fake_repo.status_marks == [{"document_id": 1, "status": "failed"}]


def test_upload_guard_rejection_deletes_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_repo: FakeRepo
) -> None:
    """Guard-stage UploadRejectedError DELETEs the placeholder outright (finding #6):
    the guards run before any store write, so a failed row would only pollute
    the inventory — no failed mark, no ghost book in the facet."""
    import tools.multimodal_ingest as mm_ingest
    import tools.node_repository as node_repo

    class MaxIdConn:
        async def fetchrow(self, query: str, *args: object):
            return {"max_id": 0}

    class MaxIdPool:
        def acquire(self):
            class _Ctx:
                async def __aenter__(self):
                    return MaxIdConn()

                async def __aexit__(self, *exc: object) -> bool:
                    return False

            return _Ctx()

    async def rejected(*args: object, **kwargs: object):
        raise mm_ingest.UploadRejectedError("PDF ch1.pdf has 900 pages (> 500); rejected")

    async def fake_pool():
        return MaxIdPool()

    monkeypatch.setattr(node_repo, "get_pool", fake_pool)
    monkeypatch.setattr(mm_ingest, "ingest_one_pdf", rejected)
    monkeypatch.setattr(document_service, "_milvus_document_exists", lambda doc_id, col: False)

    pdf = tmp_path / "ch1.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(mm_ingest.UploadRejectedError):
        asyncio.run(document_service.upload_multimodal_pdf(pdf_path=pdf, collection=None))

    assert fake_repo.deleted_placeholders == [1]
    assert fake_repo.status_marks == []


def test_resolve_upload_collection_rejects_text_library(fake_repo: FakeRepo) -> None:
    """Uploading into the SEC text collection (rag_nodes) is a 422 (review P2-2)."""
    with pytest.raises(DocumentAskError) as excinfo:
        asyncio.run(document_service.resolve_upload_collection(config.milvus_collection))
    assert "多模态上传仅支持多模态 collection" in str(excinfo.value)
    # Multimodal aliases still resolve.
    resolved = asyncio.run(document_service.resolve_upload_collection("multimodal"))
    assert resolved == config.multimodal_collection


# ── documents inventory + delete (§8.5.2) ────────────────────────────────


def test_documents_list_is_bare_array(client: TestClient, fake_repo: FakeRepo) -> None:
    fake_repo.overview = [
        {
            "document_id": 9802,
            "title": None,
            "source_uri": "C:/books/ch2.pdf",
            "metadata": {
                "source": "multimodal_pdf",
                "pages": 10,
                "chunks": {"text": 20, "image": 4},
                "book_id": "Flink",
                "chapter_label": "Flink ch2",
            },
            "created_at": "2026-09-20T00:00:01+00:00",
        },
        {
            "document_id": 9801,
            "title": "t1",
            "source_uri": "C:/books/ch1.pdf",
            "metadata": {
                "source": "multimodal_pdf",
                "pages": 8,
                "chunks": {"text": 15, "image": 1},
                "book_id": "Flink",
                "chapter_label": "Flink ch1",
            },
            "created_at": "2026-09-20T00:00:00+00:00",
        },
    ]
    response = client.get("/api/documents/documents")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)  # bare array, NOT {items: [...]}
    assert [row["document_id"] for row in body] == [9802, 9801]  # document_id DESC
    assert body[0] == {
        "document_id": 9802,
        "status": "completed",
        "filename": "ch2.pdf",
        "title": "Flink ch2",
        "page_count": 10,
        "node_count": 24,
        "book_id": "Flink",
        "chapter_label": "Flink ch2",
        "created_at": "2026-09-20T00:00:01+00:00",
    }


def test_documents_list_failure_is_502(
    client: TestClient, fake_repo: FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenRepo:
        async def multimodal_document_overview(self) -> list[dict]:
            raise RuntimeError("pg down")

    monkeypatch.setattr(document_service, "repo", BrokenRepo())
    response = client.get("/api/documents/documents")
    assert response.status_code == 502


def test_delete_document_404(client: TestClient, fake_repo: FakeRepo) -> None:
    fake_repo.exists = False
    response = client.delete("/api/documents/documents/999999")
    assert response.status_code == 404


def test_delete_document_cascade_failure_is_502_with_step(
    client: TestClient, fake_repo: FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(document_id: int):
        raise DocumentCascadeError("Milvus delete failed", step="milvus")

    monkeypatch.setattr(document_api, "delete_document", boom)
    response = client.delete("/api/documents/documents/9801")
    assert response.status_code == 502
    # Same {"message": ...} shape as the other endpoints (review P2-4).
    assert response.json()["detail"] == {"message": "cascade failed at milvus"}


def test_delete_document_success(
    client: TestClient, fake_repo: FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def ok(document_id: int):
        return {"document_id": document_id}

    monkeypatch.setattr(document_api, "delete_document", ok)
    response = client.delete("/api/documents/documents/9801")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}


# ── collections: kind field + creation (§8.5.3) ──────────────────────────


def test_collections_lists_fixed_and_dynamic_with_kind(
    client: TestClient, fake_repo: FakeRepo
) -> None:
    fake_repo.dynamic = [
        {
            "collection_name": "flink_course",
            "embedding_provider": "ark",
            "description": None,
            "created_at": "2026-09-20T00:00:00+00:00",
        }
    ]
    body = client.get("/api/documents/collections").json()
    entries = body["collections"]
    by_id = {e["id"]: e for e in entries}
    # fixed ids unchanged + kind added
    assert by_id["text"]["kind"] == "fixed"
    assert by_id["multimodal"]["name"] == config.multimodal_collection
    assert by_id["multimodal"]["kind"] == "fixed"
    # dynamic entry: id = collection_name, unavailable → points null
    assert by_id["flink_course"]["kind"] == "dynamic"
    assert by_id["flink_course"]["available"] is False
    assert by_id["flink_course"]["points"] is None
    assert {"id", "available", "points"} <= set(entries[0].keys())  # old contract intact


def test_create_collection_validation_matrix(
    client: TestClient, fake_repo: FakeRepo
) -> None:
    fake_repo.dynamic = [
        {
            "collection_name": "flink_course",
            "embedding_provider": None,
            "description": None,
            "created_at": "2026-09-20T00:00:00+00:00",
        }
    ]
    for payload, why in (
        ({"name": "Bad-Name"}, "regex"),
        ({"name": "ab"}, "too short"),
        ({"name": "multimodal"}, "fixed alias"),
        ({"name": config.milvus_collection}, "fixed text store"),
        ({"name": config.multimodal_collection}, "fixed mm store"),
        ({"name": "flink_course"}, "already registered"),
    ):
        response = client.post("/api/documents/collections", json=payload)
        assert response.status_code == 422, f"{why}: {payload}"


def test_create_collection_happy_path_registers_after_milvus(
    client: TestClient, fake_repo: FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "multimodal_embedding_dim", 3072)
    response = client.post(
        "/api/documents/collections",
        json={"name": "flink_course", "embedding_provider": "ark", "description": "课件库"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["collection_name"] == "flink_course"
    assert body["embedding_provider"] == "ark"
    assert FakeMmBackend.created == ["flink_course"]  # Milvus created BEFORE registering
    assert len(fake_repo.dynamic) == 1


def test_create_collection_milvus_failure_502_without_registering(
    client: TestClient, fake_repo: FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "multimodal_embedding_dim", 3072)
    FakeMmBackend.fail_ensure = True
    response = client.post("/api/documents/collections", json={"name": "broken_col"})
    assert response.status_code == 502
    assert fake_repo.dynamic == []  # no phantom registry entry


def test_create_collection_unknown_dim_is_422(
    client: TestClient, fake_repo: FakeRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "multimodal_embedding_dim", 0)
    response = client.post("/api/documents/collections", json={"name": "flink_course"})
    assert response.status_code == 422


# ── ask collection routing (§8.5.3) ──────────────────────────────────────


def test_ask_unknown_collection_is_422(client: TestClient, fake_repo: FakeRepo) -> None:
    response = client.post(
        "/api/documents/ask", json={"question": "什么是状态后端?", "collection": "nope"}
    )
    assert response.status_code == 422


def test_ask_routes_dynamic_collection(client: TestClient, fake_repo: FakeRepo) -> None:
    fake_repo.dynamic = [
        {
            "collection_name": "flink_course",
            "embedding_provider": None,
            "description": None,
            "created_at": "2026-09-20T00:00:00+00:00",
        }
    ]
    response = client.post(
        "/api/documents/ask", json={"question": "什么是状态后端?", "collection": "flink_course"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["collection"] == "flink_course"
    assert body["evidence"][0]["chunk_id"] == "c1"


def test_ask_text_path_unaffected(client: TestClient, fake_repo: FakeRepo) -> None:
    import tools.llamaindex_retrieval as lr
    import tools.node_repository as nr

    class FakeRetrieval:
        async def retrieve(self, query: str, document_ids):
            return {"nodes": []}

    async def fake_ids(*, limit: int = 500):
        return [1]

    fake_repo.exists = True
    monkey_lr = lr.retrieval_service
    monkey_nr = nr.list_available_document_ids
    lr.retrieval_service = FakeRetrieval()
    nr.list_available_document_ids = fake_ids
    try:
        response = client.post(
            "/api/documents/ask", json={"question": "why?", "collection": "text"}
        )
    finally:
        lr.retrieval_service = monkey_lr
        nr.list_available_document_ids = monkey_nr
    assert response.status_code == 200
    assert response.json()["collection"] == "text"


# ── testset / evaluate jobs (§8.5.4) ─────────────────────────────────────


@pytest.fixture()
def eval_whitelist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "multimodal_eval"
    target.mkdir()
    monkeypatch.setattr(config, "multimodal_eval_dir", str(target))
    return target


def test_generate_testset_job_lifecycle(
    client: TestClient, eval_whitelist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.gen_multimodal_evalset as gen

    payload = {
        "version": "t2.5-draft-1",
        "counts": {"total": 1, "single_hop": 1, "aggregation": 0, "cross_book": 0, "leak_suspect": 0},
        "questions": [
            {
                "type": "single_hop",
                "question": "q?",
                "reference": "a",
                "gold_chunk_ids": ["c1"],
                "scope": {"book_id": "b", "chapter_document_id": 1},
            }
        ],
    }

    async def fake_core(**kwargs):
        assert kwargs["total"] == 3
        return payload

    monkeypatch.setattr(gen, "generate_evalset_core", fake_core)
    response = client.post("/api/documents/generate-testset", json={"testset_size": 3})
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    settled = _wait_job(client, job_id)
    assert settled["status"] == "completed"
    assert settled["questions_count"] == 1
    assert settled["questions"] == payload["questions"]
    assert Path(settled["testset_path"]).is_file()  # written into the whitelist dir
    assert Path(settled["testset_path"]).parent == eval_whitelist


def test_generate_testset_failure_records_error(
    client: TestClient, eval_whitelist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.gen_multimodal_evalset as gen

    async def boom(**kwargs):
        raise ValueError("no chunks found; run ingest_multimodal_pdf.py first")

    monkeypatch.setattr(gen, "generate_evalset_core", boom)
    response = client.post("/api/documents/generate-testset", json={"testset_size": 3})
    assert response.status_code == 202
    settled = _wait_job(client, response.json()["job_id"])
    assert settled["status"] == "failed"
    assert "no chunks found" in settled["error"]


def test_generate_testset_size_validated(client: TestClient) -> None:
    assert client.post("/api/documents/generate-testset", json={"testset_size": 0}).status_code == 422
    assert client.post("/api/documents/generate-testset", json={"testset_size": 101}).status_code == 422


def test_get_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/documents/testset/does-not-exist").status_code == 404


def test_evaluate_whitelist_guard(client: TestClient, eval_whitelist: Path) -> None:
    outside = client.post("/api/documents/evaluate", json={"testset_path": "C:/Windows/system32/x.json"})
    assert outside.status_code == 422
    escape = client.post(
        "/api/documents/evaluate", json={"testset_path": str(eval_whitelist / "../../secrets.json")}
    )
    assert escape.status_code == 422
    missing = client.post(
        "/api/documents/evaluate", json={"testset_path": str(eval_whitelist / "absent.json")}
    )
    assert missing.status_code == 422


def test_evaluate_job_returns_report(
    client: TestClient, eval_whitelist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_multimodal_eval as runner

    testset = eval_whitelist / "draft.json"
    testset.write_text(json.dumps({"questions": []}), encoding="utf-8")
    report = {
        "generated_at": "2026-09-20T00:00:00",
        "evalset": str(testset),
        "top_k": 8,
        "summary": {"overall": {"hit_rate": 1.0, "gold_recall": 1.0, "mrr": 1.0}},
        "results": [],
    }

    async def fake_core(**kwargs):
        assert Path(kwargs["evalset"]) == testset
        return report

    monkeypatch.setattr(runner, "run_evaluation_core", fake_core)
    response = client.post(
        "/api/documents/evaluate", json={"testset_path": str(testset), "collection": "multimodal"}
    )
    assert response.status_code == 202
    settled = _wait_job(client, response.json()["job_id"])
    assert settled["status"] == "completed"
    # 与 run_multimodal_eval 报告同构
    assert settled["report"]["summary"]["overall"]["hit_rate"] == 1.0
    assert settled["report"]["top_k"] == 8
    assert Path(settled["report_path"]).parent == eval_whitelist / "reports"


# ── chapter pagination guard (review A4) ─────────────────────────────────


def test_chapter_page_beyond_milvus_window_is_422(client: TestClient) -> None:
    # offset+limit > 16384 must be rejected client-side, not 502 from Milvus.
    response = client.get("/api/documents/chapters/9801/chunks", params={"page": 200, "page_size": 100})
    assert response.status_code == 422


def test_normalize_missing_type_is_dict() -> None:
    # review A5: DocumentAskError.missing is a dict (books/chapters), not a list.
    exc = DocumentAskError("typo guard", missing={"books": ["x"], "chapters": []})
    assert exc.missing == {"books": ["x"], "chapters": []}
