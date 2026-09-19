"""Unit tests for T2.1/T2.3 — set payloads, filter aggregation, normalization, CLI helpers.

PG-dependent paths run against a fake pool (no server); the pure functions
(aggregate_filters, normalize helpers, natural sort, book meta) run directly.
"""

import asyncio

import pytest
from tools.documents import document_repository as repo
from tools.documents.document_repository import (
    MAX_ENUM_CHUNKS,
    DocumentSetError,
    aggregate_filters,
)

# ── pure: aggregate_filters (§8 filters facet) ───────────────────────────


def _doc(doc_id: int, book: str, chapter: int, label: str, text: int = 3, image: int = 1, uri: str | None = None) -> dict:
    return {
        "document_id": doc_id,
        "title": label,
        "source_uri": uri or f"C:/books/{label}.pdf",
        "metadata": {
            "source": "multimodal_pdf",
            "book_id": book,
            "chapter_index": chapter,
            "chapter_label": label,
            "pages": 14,
            "chunks": {"text": text, "image": image},
        },
    }


def test_aggregate_filters_groups_books_and_sorts_chapters():
    documents = [
        _doc(2, "Apache Flink", 2, "第二章"),
        _doc(1, "Apache Flink", 1, "第一章"),
        _doc(3, "PEFT 论文", 1, "Parameter-Efficient"),
    ]
    filters = aggregate_filters(documents)
    assert [b["book_id"] for b in filters["books"]] == ["Apache Flink", "PEFT 论文"]
    flink = filters["books"][0]
    assert flink["chapter_count"] == 2
    assert [c["chapter_index"] for c in flink["chapters"]] == [1, 2]
    assert flink["chunk_count"] == (3 + 1) * 2
    assert filters["kinds"] == ["image", "text"]
    assert flink["chapters"][0]["filename"] == "第一章.pdf"


def test_aggregate_filters_empty():
    assert aggregate_filters([]) == {"books": [], "kinds": set()} or aggregate_filters([])["books"] == []


# ── pure: CLI helpers (natural sort + book meta, §14.1-6/7) ──────────────

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "agent" / "scripts"))

from ingest_multimodal_pdf import build_book_meta, natural_sort_key  # noqa: E402


def test_natural_sort_ch2_before_ch10():
    names = ["ch10.pdf", "ch2.pdf", "ch1.pdf", "第一章.pdf"]
    ordered = sorted(names, key=natural_sort_key)
    assert ordered == ["ch1.pdf", "ch2.pdf", "ch10.pdf", "第一章.pdf"]


def test_build_book_meta_with_book():
    files = [Path("ch1.pdf"), Path("ch2.pdf")]
    metas = build_book_meta("Apache Flink", 2, files)
    assert metas[0] == {
        "book_id": "Apache Flink",
        "chapter_index": 2,
        "chapter_label": "Apache Flink · 第2章 · ch1",
    }
    assert metas[1]["chapter_index"] == 3


def test_build_book_meta_without_book_is_own_book():
    metas = build_book_meta(None, 1, [Path("第一章 Apache Flink 概述.pdf")])
    assert metas[0]["book_id"] == "第一章 Apache Flink 概述"
    assert metas[0]["chapter_index"] == 1


# ── set payload validation (pure part of T2.1) ───────────────────────────


def test_validate_set_payload_enum_cap():
    ids = [f"id{i}" for i in range(MAX_ENUM_CHUNKS + 1)]
    with pytest.raises(DocumentSetError, match="cap"):
        repo._validate_set_payload("big", "enumerated", None, ids)


def test_validate_set_payload_rejects_bad_kind():
    with pytest.raises(DocumentSetError, match="kind"):
        repo._validate_set_payload("x", "magic", None, None)


def test_validate_set_payload_filter_requires_content():
    with pytest.raises(DocumentSetError, match="filter"):
        repo._validate_set_payload("x", "filter", {}, None)
    name, fj, ci = repo._validate_set_payload("x", "filter", {"books": ["b"]}, None)
    assert name == "x" and fj == {"books": ["b"]} and ci is None


# ── normalize_filters against a fake repository (§15 error matrix) ───────


class FakeRepo:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs

    async def get_set(self, set_id: str):
        for found in self.sets:
            if found["set_id"] == set_id:
                return found
        return None

    sets: list[dict] = []

    async def list_multimodal_documents(self):
        return self.docs

    async def expand_books_to_doc_ids(self, book_ids):
        wanted = set(book_ids)
        found, seen = [], set()
        for d in self.docs:
            if d["metadata"]["book_id"] in wanted:
                seen.add(d["metadata"]["book_id"])
                found.append(d["document_id"])
        return sorted(set(found)), sorted(wanted - seen)

    async def expand_chapters(self, document_ids):
        known = {d["document_id"] for d in self.docs}
        return [i for i in document_ids if i in known], [i for i in document_ids if i not in known]


DOCS = [
    _doc(1, "Apache Flink", 1, "第一章"),
    _doc(2, "Apache Flink", 2, "第二章"),
    _doc(3, "PEFT", 1, "论文"),
]


@pytest.fixture()
def patched_repo(monkeypatch: pytest.MonkeyPatch):
    fake = FakeRepo(DOCS)
    monkeypatch.setattr("tools.documents.document_service.repo", fake)
    return fake


def _norm(filters=None, set_id=None):
    from tools.documents.document_service import normalize_filters

    return asyncio.run(normalize_filters(filters, set_id))


def test_normalize_books_and_chapters_union(patched_repo):
    out = _norm({"books": ["Apache Flink"], "chapters": [3]})
    assert out["document_ids"] == [1, 2, 3]  # union, no intersection trap
    assert out["chunk_ids"] is None


def test_normalize_kinds_validated(patched_repo):
    out = _norm({"kinds": ["image"]})
    assert out["kinds"] == ["image"]
    with pytest.raises(Exception, match="invalid kinds"):
        _norm({"kinds": ["video"]})


def test_normalize_unknown_book_is_422(patched_repo):
    from tools.documents.document_service import DocumentAskError

    with pytest.raises(DocumentAskError) as exc_info:
        _norm({"books": ["不存在的书"]})
    assert exc_info.value.status_code == 422
    assert exc_info.value.missing["books"] == ["不存在的书"]


def test_normalize_unknown_chapter_alone_is_422(patched_repo):
    from tools.documents.document_service import DocumentAskError

    with pytest.raises(DocumentAskError):
        _norm({"chapters": [999]})


def test_filters_and_set_id_mutually_exclusive(patched_repo):
    from tools.documents.document_service import DocumentAskError

    with pytest.raises(DocumentAskError):
        _norm({"books": ["Apache Flink"]}, set_id="s1")


def test_set_id_enumerated_expands_chunk_ids(patched_repo):
    patched_repo.sets = [
        {"set_id": "s1", "name": "n", "kind": "enumerated", "filter_json": None, "chunk_ids": ["a", "b"]}
    ]
    out = _norm(set_id="s1")
    assert out["chunk_ids"] == ["a", "b"]


def test_set_id_filter_kind_expands_through_same_path(patched_repo):
    patched_repo.sets = [
        {"set_id": "s2", "name": "n", "kind": "filter", "filter_json": {"books": ["PEFT"]}, "chunk_ids": None}
    ]
    out = _norm(set_id="s2")
    assert out["document_ids"] == [3]


def test_set_id_missing_is_404(patched_repo):
    from tools.documents.document_service import DocumentAskError

    with pytest.raises(DocumentAskError) as exc_info:
        _norm(set_id="nope")
    assert exc_info.value.status_code == 404
