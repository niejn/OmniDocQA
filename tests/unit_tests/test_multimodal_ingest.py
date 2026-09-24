"""P1-4 / P1-1 / P2-1 / P2-11 regression tests for tools/multimodal_ingest.py.

- page-count pre-check rejects a huge PDF BEFORE full rasterization (OOM guard);
- guard errors raise UploadRejectedError (ValueError subclass → CLI contract
  unchanged, API 422-by-type instead of substring guessing);
- sync pipeline segments run through worker threads (function stays async);
- PG registration failure triggers a best-effort Milvus/asset rollback.

Everything is faked offline except fitz building the tiny fixture PDFs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import tools.node_repository as node_repo
import tools.retrieval_backends.dense_milvus_multimodal as mm_backend_mod
from core.config import config
from tools.dots_ocr_client import ParsedPage
from tools.multimodal_ingest import UploadRejectedError, build_book_meta, ingest_one_pdf


def _write_pdf(path: Path, pages: int) -> Path:
    import fitz

    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(str(path))
    doc.close()
    return path


def _no_rasterize(*_args: object, **_kwargs: object):
    raise AssertionError("load_page_images must not run before the page-count pre-check")


class FakeClient:
    """DotsOcrClient stand-in: healthy, no network, no thread pool."""

    calls: list[str] = []

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or "http://fake"

    def healthy(self, timeout: float = 3.0) -> bool:
        return True

    def parse_pdf(self, pdf_path, page_images=None):
        FakeClient.calls.append("parse_pdf")
        return [
            ParsedPage(page_no=0, md_content="# 第一章\n\n短文本。"),
            ParsedPage(page_no=1, md_content="## 第二节\n\n更多文本。"),
        ]


@pytest.fixture(autouse=True)
def _reset():
    FakeClient.calls = []
    yield


# ── P1-4: page-count pre-check before rasterization ──────────────────────


def test_page_cap_rejects_before_rasterizing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.dots_ocr_client as dots

    pdf = _write_pdf(tmp_path / "big.pdf", pages=3)
    monkeypatch.setattr(config, "multimodal_upload_max_pages", 2)
    monkeypatch.setattr(dots, "load_page_images", _no_rasterize)

    with pytest.raises(UploadRejectedError) as excinfo:
        asyncio.run(
            ingest_one_pdf(pdf, 1, build_book_meta(None, 1, [pdf])[0], dry_run=True)
        )
    assert "has 3 pages (> 2)" in str(excinfo.value)


def test_oversize_rejected_before_any_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.dots_ocr_client as dots

    pdf = tmp_path / "fat.pdf"
    pdf.write_bytes(b"%PDF-1.4" + b"x" * (1024 * 1024))
    monkeypatch.setattr(config, "multimodal_upload_max_mb", 1)
    monkeypatch.setattr(dots, "load_page_images", _no_rasterize)

    with pytest.raises(UploadRejectedError) as excinfo:
        asyncio.run(
            ingest_one_pdf(pdf, 1, build_book_meta(None, 1, [pdf])[0], dry_run=True)
        )
    assert "bytes" in str(excinfo.value)


# ── P2-1: exception-type contract ────────────────────────────────────────


def test_upload_rejected_error_is_value_error_subclass() -> None:
    # CLI `except ValueError` semantics unchanged; plain ValueErrors (dim/
    # collection/embedding family) must NOT be classified as upload rejections.
    assert issubclass(UploadRejectedError, ValueError)
    assert not isinstance(ValueError("dim=1536 != embedding dim=3072"), UploadRejectedError)


def test_chunking_vectors_aborts_on_quota_exhausted() -> None:
    # Live 09-23 finding: swallowing quota failures made the chunker grind
    # through every batch (~70ms/call) while the upload request hung.
    from tools.multimodal_ingest import _chunking_vectors
    from tools.multimodal_vectorizer import EmbeddingQuotaExceededError, EmbedResult

    results = [
        EmbedResult(vector=[0.1]),
        EmbedResult(
            vector=None,
            quota_exhausted=True,
            error="embedding provider quota exhausted (ark AccountQuotaExceeded): reset at 2026-10-31 23:59:59 +0800 CST.",
        ),
    ]
    with pytest.raises(EmbeddingQuotaExceededError, match="AccountQuotaExceeded"):
        _chunking_vectors(results)


def test_chunking_vectors_all_failed_raises() -> None:
    from tools.multimodal_ingest import _chunking_vectors
    from tools.multimodal_vectorizer import EmbedResult

    with pytest.raises(ValueError, match="chunking embeddings all failed"):
        _chunking_vectors([EmbedResult(vector=None, error="HTTP 500: boom")])


def test_chunking_vectors_partial_success_filters() -> None:
    from tools.multimodal_ingest import _chunking_vectors
    from tools.multimodal_vectorizer import EmbedResult

    mixed = [
        EmbedResult(vector=[0.1]),
        EmbedResult(vector=None, error="HTTP 400: image too small"),
        EmbedResult(vector=[0.2]),
    ]
    assert _chunking_vectors(mixed) == [[0.1], [0.2]]
    assert _chunking_vectors([]) == []


# ── P1-1: async surface + worker-thread sync segments (dry-run path) ─────


def test_dry_run_happy_path_stays_async_with_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.dots_ocr_client as dots

    pdf = _write_pdf(tmp_path / "book.pdf", pages=2)
    monkeypatch.setattr(dots, "DotsOcrClient", FakeClient)

    def fake_load(pdf_path, dpi=None):
        FakeClient.calls.append("load_page_images")
        return [object(), object()], [(10, 10), (10, 10)]

    monkeypatch.setattr(dots, "load_page_images", fake_load)

    summary = asyncio.run(
        ingest_one_pdf(pdf, 1, build_book_meta(None, 1, [pdf])[0], dry_run=True)
    )

    assert FakeClient.calls == ["load_page_images", "parse_pdf"]
    assert summary["backend"] == "dots_ocr"
    assert summary["pages"] == 2
    assert summary["chunks_text"] >= 2
    assert summary["dry_run"] is True
    assert summary["estimated_calls"]["parse_pages"] == 2


# ── P2-11: PG registration failure → best-effort store rollback ──────────


class RecordingMmBackend:
    """MilvusMultimodalDenseBackend stand-in recording upsert/rollback order."""

    def __init__(self, collection: str | None = None) -> None:
        self.collection = collection or config.multimodal_collection

    async def upsert_document_nodes(self, document_id, chunks, **kwargs):
        ORDER.append("milvus_upsert")
        return {"inserted": len(chunks), "truncated": 0, "dim": 4}

    def replace_document_nodes(self, document_id, nodes=None):
        ORDER.append("milvus_rollback")

    def ensure_collection(self, vector_size=None) -> None:
        return None


class RecordingAssetStore:
    def save(self, document_id: int, name: str, data: bytes) -> str:
        return f"{document_id}/{name}"

    def document_bytes(self, document_id: int) -> int:
        return 10

    def delete_document(self, document_id: int) -> None:
        ORDER.append("assets_rollback")


ORDER: list[str] = []


@pytest.fixture(autouse=True)
def _reset_order():
    ORDER.clear()
    yield
    ORDER.clear()


class _GuardConn:
    async def fetchrow(self, query: str, *args: object):
        return None  # no existing row → overwrite guard passes


class _GuardPool:
    def acquire(self):
        class _Ctx:
            async def __aenter__(self):
                return _GuardConn()

            async def __aexit__(self, *exc: object) -> bool:
                return False

        return _Ctx()


def test_pg_registration_failure_rolls_back_milvus_and_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.dots_ocr_client as dots
    import tools.multimodal_asset_store as asset_store

    pdf = _write_pdf(tmp_path / "book.pdf", pages=1)

    async def fake_get_pool():
        return _GuardPool()

    monkeypatch.setattr(node_repo, "get_pool", fake_get_pool)
    monkeypatch.setattr(node_repo, "upsert_document", _pg_boom)
    monkeypatch.setattr(mm_backend_mod, "MilvusMultimodalDenseBackend", RecordingMmBackend)
    monkeypatch.setattr(asset_store, "get_asset_store", lambda: RecordingAssetStore())
    monkeypatch.setattr(dots, "DotsOcrClient", FakeClient)
    monkeypatch.setattr(dots, "load_page_images", lambda pdf_path, dpi=None: ([object()], [(10, 10)]))

    with pytest.raises(RuntimeError, match="pg down"):
        asyncio.run(
            ingest_one_pdf(
                pdf,
                7,
                build_book_meta("TestBook", 1, [pdf])[0],
                collection="mm_col",
                extra_metadata={"status": "completed"},
            )
        )

    # Milvus points + assets are compensated BEFORE the error surfaces.
    assert ORDER == ["milvus_upsert", "pg_upsert", "milvus_rollback", "assets_rollback"]


async def _pg_boom(*_args: object, **_kwargs: object):
    ORDER.append("pg_upsert")
    raise RuntimeError("pg down")
