"""Unit tests for the Milvus multimodal backend (T1.4) — FakeClient, no server."""

import asyncio
from typing import Any

import pytest
from core.config import config
from tools.multimodal_vectorizer import EmbedResult, MultimodalVectorizer
from tools.retrieval_backends import dense_milvus_multimodal as mm
from tools.retrieval_backends.dense_milvus_multimodal import (
    MilvusMultimodalDenseBackend,
    MmChunkRow,
    build_filter_expr,
)


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[str] = []
        self.functions: list[str] = []

    def add_field(self, field_name: str, datatype: object = None, **kwargs: object) -> None:
        self.fields.append(field_name)
        self.kwargs = kwargs

    def add_function(self, function: object) -> None:
        self.functions.append(str(getattr(function, "name", function)))


class FakeIndexParams:
    def add_index(self, *, field_name: str, **kwargs: object) -> None:
        pass


class FakeClient:
    def __init__(self) -> None:
        self.collections: set[str] = {"rag_multimodal"}
        self.created: list | None = None
        self.inserted: list[dict] = []
        self.deleted_filters: list[str] = []
        self.search_calls: list[dict] = []
        self.search_result: list = []
        self.query_calls: list[dict] = []
        self.query_rows: list[dict] = []
        self.describe_result: dict = {"fields": [{"name": "dense", "params": {"dim": 3}}]}

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_schema(self, **kwargs: object) -> FakeSchema:
        return FakeSchema()

    def prepare_index_params(self) -> FakeIndexParams:
        return FakeIndexParams()

    def create_collection(self, *, collection_name: str, schema, index_params) -> None:
        self.collections.add(collection_name)
        self.created = schema

    def describe_collection(self, *, collection_name: str) -> dict:
        return self.describe_result

    def insert(self, *, collection_name: str, data: list[dict]) -> None:
        self.inserted.extend(data)

    def delete(self, *, collection_name: str, filter: str) -> None:  # noqa: A002
        self.deleted_filters.append(filter)

    def search(self, **kwargs: object) -> list:
        self.search_calls.append(kwargs)
        return self.search_result

    def query(self, *, collection_name: str, filter: str, output_fields=None, limit=None, offset=None, **kw):  # noqa: A002
        self.query_calls.append({"filter": filter, "limit": limit, "offset": offset})
        return self.query_rows

    def get_collection_stats(self, *, collection_name: str) -> dict:
        return {"row_count": 7}


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient()
    monkeypatch.setattr(mm, "get_client", lambda: fake)
    monkeypatch.setattr(config, "multimodal_collection", "rag_multimodal")
    return fake


class FakeVectorizer(MultimodalVectorizer):
    """Record calls, return deterministic vectors without any HTTP."""

    def __init__(self, dim: int = 3, fail_on_text: str | None = None, quota_fail: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.text_calls: list[str] = []
        self.image_calls: list[str] = []
        self.fail_on_text = fail_on_text
        self.quota_fail = quota_fail

    async def embed_text(self, text: str) -> EmbedResult:
        self.text_calls.append(text)
        if self.quota_fail:
            return EmbedResult(
                vector=None,
                quota_exhausted=True,
                error="embedding provider quota exhausted (ark AccountQuotaExceeded): "
                "It will reset at 2026-09-23 23:59:59 +0800 CST.",
            )
        if self.fail_on_text and self.fail_on_text in text:
            return EmbedResult(vector=None, error="boom")
        return EmbedResult(vector=[0.1] * self.dim)

    async def embed_image_with_text(self, image_data_uri: str, text: str) -> EmbedResult:
        self.image_calls.append(image_data_uri[:30])
        return EmbedResult(vector=[0.2] * self.dim)


def _chunk(**overrides: Any) -> MmChunkRow:
    fields: dict[str, Any] = {
        "kind": "text",
        "page_no": 0,
        "title": "第一章 --> 小节",
        "text": "Flink 是流式计算框架。",
        "category": "Text",
    }
    fields.update(overrides)
    return MmChunkRow(**fields)


def test_build_filter_expr_doc_union_and_kind():
    expr = build_filter_expr(document_ids=[3, 1, 3], kinds=["text", "image"])
    assert expr == 'document_id in [1, 3] and kind in ["image", "text"]'


def test_build_filter_expr_chunk_ids_pk():
    expr = build_filter_expr(chunk_ids=["a", "b", "a"])
    assert expr == 'id in ["a", "b"]'


def test_build_filter_expr_empty():
    assert build_filter_expr() == ""


def test_upsert_embeds_inside_backend_then_replaces(fake_client: FakeClient):
    backend = MilvusMultimodalDenseBackend()
    vectorizer = FakeVectorizer()
    stats = asyncio.run(
        backend.upsert_document_nodes(
            901,
            [_chunk(), _chunk(kind="image", image_ref="901/image_p0_0.jpg", image_data_uri="data:image/jpeg;base64,AAA", text="描述")],
            vectorizer=vectorizer,
            filename="第一章.pdf",
            book_id="Apache Flink",
            chapter_label="Apache Flink · 第1章 · 第一章",
        )
    )
    assert stats["inserted"] == 2 and stats["truncated"] == 0
    assert fake_client.deleted_filters == ["document_id == 901"]  # idempotent replace
    assert len(fake_client.inserted) == 2
    text_row = fake_client.inserted[0]
    image_row = fake_client.inserted[1]
    # text chunk embedding = title-prefixed body; image chunk = image call
    assert vectorizer.text_calls == ["第一章 --> 小节：Flink 是流式计算框架。"]
    assert len(vectorizer.image_calls) == 1
    assert text_row["book_id"] == "Apache Flink"
    assert text_row["chapter_label"] == "Apache Flink · 第1章 · 第一章"
    assert text_row["kind"] == "text" and image_row["kind"] == "image"
    assert image_row["image_ref"] == "901/image_p0_0.jpg"
    assert text_row["text"].startswith("Flink 是流式计算框架。")


def test_upsert_embedding_failure_is_fail_fast(fake_client: FakeClient):
    backend = MilvusMultimodalDenseBackend()
    vectorizer = FakeVectorizer(fail_on_text="Flink")
    with pytest.raises(ValueError, match="embedding failed"):
        asyncio.run(
            backend.upsert_document_nodes(
                902, [_chunk()], vectorizer=vectorizer, filename="f.pdf", book_id="b", chapter_label="c"
            )
        )
    assert fake_client.inserted == []  # nothing written on failure
    assert fake_client.deleted_filters == []  # previous points survive (delete runs after validation)


def test_upsert_quota_exhausted_raises_typed_error(fake_client: FakeClient):
    """Quota-exhausted embedding results raise EmbeddingQuotaExceededError (the
    upload API's 503 signal), not the generic store-failure ValueError."""
    from tools.multimodal_vectorizer import EmbeddingQuotaExceededError

    backend = MilvusMultimodalDenseBackend()
    with pytest.raises(EmbeddingQuotaExceededError, match="AccountQuotaExceeded"):
        asyncio.run(
            backend.upsert_document_nodes(
                904, [_chunk()], vectorizer=FakeVectorizer(quota_fail=True), filename="f.pdf", book_id="b", chapter_label="c"
            )
        )
    assert fake_client.inserted == [] and fake_client.deleted_filters == []


def test_upsert_image_without_data_uri_raises(fake_client: FakeClient):
    backend = MilvusMultimodalDenseBackend()
    with pytest.raises(ValueError, match="image_data_uri"):
        asyncio.run(
            backend.upsert_document_nodes(
                903,
                [_chunk(kind="image", image_ref="x")],
                vectorizer=FakeVectorizer(),
                filename="f.pdf",
                book_id="b",
                chapter_label="c",
            )
        )


def test_dim_mismatch_fails_fast(fake_client: FakeClient):
    fake_client.collections = {"rag_multimodal"}
    backend = MilvusMultimodalDenseBackend()
    with pytest.raises(ValueError, match="dim"):
        backend.ensure_collection(999)  # describe says 3


def test_search_maps_hit_fields(fake_client: FakeClient):
    fake_client.search_result = [
        [
            {
                "id": "chunk-1",
                "distance": 0.87,
                "entity": {
                    "document_id": 901,
                    "filename": "第一章.pdf",
                    "title": "第一章",
                    "kind": "image",
                    "page_no": 0,
                    "text": "图片描述" + "x" * 3000,
                    "category": "Picture",
                    "image_ref": "901/image_p0_0.jpg",
                    "book_id": "Apache Flink",
                    "chapter_label": "Apache Flink · 第1章 · 第一章",
                },
            }
        ]
    ]
    backend = MilvusMultimodalDenseBackend()
    hits = backend.search([0.1] * 3, document_ids=[901], kinds=["image"], limit=5, log_stage="test")
    assert len(hits) == 1
    hit = hits[0]
    assert hit["chunk_id"] == "chunk-1"
    assert hit["score"] == pytest.approx(0.87)
    assert hit["kind"] == "image"
    assert hit["book_id"] == "Apache Flink"
    assert len(hit["text_preview"]) == 2000  # preview capped
    call = fake_client.search_calls[0]
    assert call["anns_field"] == "dense"
    assert call["filter"] == 'document_id in [901] and kind in ["image"]'
    assert call["limit"] == 5


def test_search_whole_library_no_filter(fake_client: FakeClient):
    backend = MilvusMultimodalDenseBackend()
    backend.search([0.1] * 3, limit=3)
    assert fake_client.search_calls[0]["filter"] is None


def test_count_and_existing_chunk_ids(fake_client: FakeClient):
    fake_client.query_rows = [{"id": "a"}, {"id": "b"}]
    backend = MilvusMultimodalDenseBackend()
    assert backend.count() == 7
    assert backend.existing_chunk_ids(["a", "b", "c"]) == {"a", "b"}


def test_protocol_signature_compatible(fake_client: FakeClient):
    backend = MilvusMultimodalDenseBackend()
    # DenseBackend protocol: replace_document_nodes(document_id, nodes) callable form
    backend.replace_document_nodes(1, [])
    assert fake_client.deleted_filters == ["document_id == 1"]
