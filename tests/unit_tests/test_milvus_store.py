"""Unit tests for the Milvus dense backend (Step 1A, Qdrant parity).

All tests run against a fake MilvusClient — no server required. The fake
mirrors the pymilvus MilvusClient surface used by tools/milvus_store.py:
create_schema/prepare_index_params/create_collection/has_collection/insert/
delete/search, with search returning the same ``[{id, distance, entity}]``
hit shape as pymilvus.
"""

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from core.config import config
from loguru import logger
from tools import milvus_store
from tools.retrieval_backends import factory
from tools.retrieval_backends.dense_milvus import MilvusDenseBackend
from tools.retrieval_backends.sparse_milvus import MilvusSparseBackend


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[str] = []
        self.functions: list[str] = []

    def add_field(self, field_name: str, datatype: object = None, **kwargs: object) -> None:
        self.fields.append(field_name)

    def add_function(self, function: object) -> None:
        self.functions.append(str(getattr(function, "name", function)))


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[str] = []

    def add_index(self, *, field_name: str, **kwargs: object) -> None:
        self.indexes.append(field_name)


class FakeClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.create_collection_calls = 0
        self.created_schema: FakeSchema | None = None
        self.created_index: FakeIndexParams | None = None
        self.inserted: list[dict] = []
        self.deleted_filters: list[str] = []
        self.search_calls: list[dict] = []
        self.search_result: list = []
        self.upserted: list[dict] = []
        self.query_calls: list[dict] = []
        self.query_rows: list[dict] = []
        self.probe_result: list[dict] = []

    def query(self, *, collection_name: str, filter: str, output_fields: list, limit: int) -> list:  # noqa: A002
        self.query_calls.append({"filter": filter, "output_fields": list(output_fields or []), "limit": limit})
        if filter == "document_id > 0":
            return self.probe_result
        return self.query_rows

    def upsert(self, *, collection_name: str, data: list[dict]) -> None:
        self.upserted.extend(data)

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_schema(self, **kwargs: object) -> FakeSchema:
        return FakeSchema()

    def prepare_index_params(self) -> FakeIndexParams:
        return FakeIndexParams()

    def create_collection(self, *, collection_name: str, schema: FakeSchema, index_params: FakeIndexParams) -> None:
        self.create_collection_calls += 1
        self.collections.add(collection_name)
        self.created_schema = schema
        self.created_index = index_params

    def insert(self, *, collection_name: str, data: list[dict]) -> None:
        assert collection_name in self.collections, "insert before ensure_collection"
        self.inserted.extend(data)

    def delete(self, *, collection_name: str, filter: str) -> None:
        self.deleted_filters.append(filter)

    def search(self, **kwargs: object) -> list:
        self.search_calls.append(kwargs)
        return self.search_result


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient()
    monkeypatch.setattr(milvus_store, "get_client", lambda: fake)
    monkeypatch.setattr(config, "milvus_collection", "test_rag_nodes")
    return fake


def _node(**overrides: object) -> dict[str, object]:
    node: dict[str, object] = {
        "node_id": "11111111-1111-1111-1111-111111111111",
        "document_id": 9,
        "parent_id": "22222222-2222-2222-2222-222222222222",
        "node_type": "chunk",
        "level": 0,
        "order_index": 3,
        "title": "Business",
        "text": "Apple Inc. revenue discussion.",
        "metadata": {"section_path": ["Business"]},
        "vector": [0.1, 0.2, 0.3],
        "domain": "finance",
        "finance_accns": ["0000320193-25-000006", "0000320193-24-000123"],
        "topic_tags": [],
    }
    node.update(overrides)
    return node


def test_ensure_collection_creates_full_schema_once(fake_client: FakeClient) -> None:
    milvus_store.ensure_collection(3)

    assert fake_client.create_collection_calls == 1
    assert "test_rag_nodes" in fake_client.collections
    assert fake_client.created_schema is not None
    assert set(fake_client.created_schema.fields) == {
        "id",
        "document_id",
        "parent_id",
        "node_type",
        "level",
        "order_index",
        "title",
        "text",
        "metadata",
        "retrieval_fields",
        "sparse",
        "dense",
    }
    # BM25 function wired for the Step 1B sparse path (schema built full).
    assert fake_client.created_schema.functions == ["text_bm25_emb"]
    assert fake_client.created_index is not None
    assert set(fake_client.created_index.indexes) == {"dense", "sparse", "document_id"}

    # Idempotent: second call is a no-op.
    milvus_store.ensure_collection(3)
    assert fake_client.create_collection_calls == 1


def test_node_to_row_normalizes_retrieval_fields_to_string_arrays() -> None:
    row = milvus_store._node_to_row(
        _node(
            domain="finance",
            search_hints="revenue income fiscal year",  # text field: must NOT land in retrieval_fields
            section_path_text="Business",
        )
    )
    assert row["retrieval_fields"]["domain"] == ["finance"]
    assert row["retrieval_fields"]["finance_accns"] == [
        "0000320193-25-000006",
        "0000320193-24-000123",
    ]
    # Empty values are dropped; text fields are excluded.
    assert "topic_tags" not in row["retrieval_fields"]
    assert "search_hints" not in row["retrieval_fields"]
    assert "section_path_text" not in row["retrieval_fields"]


def test_node_to_row_truncates_and_casts() -> None:
    row = milvus_store._node_to_row(
        _node(
            node_id="uuid-1",
            document_id="42",
            text="x" * 70_000,
            title="t" * 1_500,
            parent_id=None,
            level="0",
        )
    )
    assert row["id"] == "uuid-1"
    assert row["document_id"] == 42
    assert row["level"] == 0
    assert row["parent_id"] is None
    assert len(row["text"]) == 65_000
    assert len(row["title"]) == 1_000


def test_node_to_row_rejects_bool_document_id() -> None:
    with pytest.raises(TypeError):
        milvus_store._node_to_row(_node(document_id=True))


def test_build_filter_expr_full_parity() -> None:
    expr = milvus_store.build_filter_expr(
        [9, 1],
        levels=[2, 0, 2],
        parent_ids=["p-2", "p-1"],
        metadata_filters={"domain": ["finance", "generic"], "finance_accns": ["0001"]},
    )
    assert expr == (
        "document_id in [9, 1] and level in [0, 2] and parent_id in [\"p-1\", \"p-2\"] "
        'and json_contains_any(retrieval_fields["domain"], ["finance", "generic"]) '
        'and json_contains_any(retrieval_fields["finance_accns"], ["0001"])'
    )


def test_build_filter_expr_escapes_and_skips_invalid_values() -> None:
    expr = milvus_store.build_filter_expr(
        cast(list[int], ["abc", 7]),
        metadata_filters={"domain": ['say "hi"', " "]},
    )
    assert expr == 'document_id in [7] and json_contains_any(retrieval_fields["domain"], ["say \\"hi\\""])'

    assert milvus_store.build_filter_expr(cast(list[int], ["nope", "bad"])) == ""


def test_dense_search_empty_inputs_return_empty_without_client_calls(fake_client: FakeClient) -> None:
    assert milvus_store.dense_search([0.1, 0.2], document_ids=[], limit=5, log_stage="unit") == []
    assert milvus_store.dense_search([], document_ids=[9], limit=5, log_stage="unit") == []
    assert milvus_store.dense_search([0.1], document_ids=cast(list[int], ["bad"]), limit=5) == []
    assert fake_client.search_calls == []


def test_dense_search_maps_hits_and_flattens_retrieval_fields(fake_client: FakeClient) -> None:
    fake_client.search_result = [
        [
            {
                "id": "uuid-1",
                "distance": 0.83,
                "entity": {
                    "document_id": 9,
                    "parent_id": "p-1",
                    "node_type": "chunk",
                    "level": 0,
                    "order_index": 3,
                    "title": "Business",
                    "metadata": {"section_path": ["Business"]},
                    "retrieval_fields": {"domain": ["finance"]},
                    "text": "y" * 1_500,
                },
            }
        ]
    ]
    results = milvus_store.dense_search(
        [0.1, 0.2],
        document_ids=[9],
        limit=5,
        levels=[0],
        metadata_filters={"domain": ["finance"]},
    )
    assert len(results) == 1
    hit = results[0]
    assert hit["node_id"] == "uuid-1"
    assert hit["dense_score"] == 0.83
    assert hit["document_id"] == 9
    assert hit["parent_id"] == "p-1"
    assert hit["title"] == "Business"
    assert len(hit["text_preview"]) == 1_000
    # Retrieval fields are flattened to the top level (Qdrant payload parity).
    assert hit["domain"] == ["finance"]

    call = fake_client.search_calls[0]
    assert call["anns_field"] == "dense"
    assert call["limit"] == 5
    assert 'document_id in [9]' in call["filter"]
    assert 'json_contains_any(retrieval_fields["domain"], ["finance"])' in call["filter"]
    assert "text" in call["output_fields"]


def test_backend_replace_document_nodes_deletes_then_inserts(fake_client: FakeClient) -> None:
    backend = MilvusDenseBackend()
    backend.replace_document_nodes(9, [_node()])
    # First write on a fresh collection: nothing to delete yet (guarded no-op).
    assert fake_client.deleted_filters == []
    assert len(fake_client.inserted) == 1
    assert fake_client.inserted[0]["id"] == "11111111-1111-1111-1111-111111111111"
    assert fake_client.inserted[0]["document_id"] == 9
    # BM25 input text is written from day one (schema built full for Step 1B),
    # now enriched with title repeats (search_hints absent in this fixture).
    assert fake_client.inserted[0]["text"] == "Business\nBusiness\nApple Inc. revenue discussion."

    # Re-replace with no nodes: delete still runs, insert does not (idempotent clear).
    backend.replace_document_nodes(9, [])
    # Collection now exists: the delete-then-insert contract holds on re-runs.
    assert fake_client.deleted_filters == ["document_id == 9"]
    assert len(fake_client.inserted) == 1


def test_delete_document_nodes_is_noop_without_collection(fake_client: FakeClient) -> None:
    milvus_store.delete_document_nodes(9)
    assert fake_client.deleted_filters == []


def test_factory_selects_backend_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    factory.get_dense_backend.cache_clear()
    monkeypatch.setattr(config, "dense_backend", "milvus")
    assert isinstance(factory.get_dense_backend(), MilvusDenseBackend)

    factory.get_dense_backend.cache_clear()
    monkeypatch.setattr(config, "dense_backend", "qdrant")
    with pytest.raises(ValueError, match="qdrant was removed at M5"):
        factory.get_dense_backend()

    factory.get_dense_backend.cache_clear()
    monkeypatch.setattr(config, "dense_backend", "bogus")
    with pytest.raises(ValueError, match="Unsupported dense backend"):
        factory.get_dense_backend()
    factory.get_dense_backend.cache_clear()


def test_sparse_search_sends_query_text_over_sparse_field(fake_client: FakeClient) -> None:
    fake_client.search_result = [
        [
            {
                "id": "uuid-s1",
                "distance": 12.5,
                "entity": {
                    "document_id": 9,
                    "parent_id": "p-1",
                    "node_type": "chunk",
                    "level": 0,
                    "order_index": 7,
                    "title": "Net Sales",
                    "metadata": {"section_path": ["Item 7"]},
                    "retrieval_fields": {"domain": ["finance"]},
                    "text": "y" * 1_500,
                },
            }
        ]
    ]
    results = milvus_store.sparse_search(
        "  net sales iPhone  ",
        document_ids=[9],
        limit=5,
        levels=[0],
        metadata_filters={"domain": ["finance"]},
    )
    assert len(results) == 1
    hit = results[0]
    assert hit["node_id"] == "uuid-s1"
    assert hit["sparse_score"] == 12.5
    assert hit["document_id"] == 9
    assert hit["level"] == 0
    assert len(hit["text"]) == 1_500
    assert len(hit["text_preview"]) == 1_000
    assert hit["domain"] == ["finance"]

    call = fake_client.search_calls[0]
    assert call["anns_field"] == "sparse"
    assert call["search_params"] == {"metric_type": "BM25"}
    assert call["data"] == ["net sales iPhone"]  # trimmed raw text, analyzed server-side
    assert call["limit"] == 5
    assert 'document_id in [9]' in call["filter"]
    assert 'level in [0]' in call["filter"]
    assert "text" in call["output_fields"]


def test_sparse_search_with_query_plan_executes_and_marks_plan_unapplied(
    fake_client: FakeClient,
) -> None:
    fake_client.search_result = [[]]
    plan = SimpleNamespace(profile="finance_v1")
    logged: list[str] = []
    handler_id = logger.add(lambda message: logged.append(message), level="INFO")
    try:
        results = milvus_store.sparse_search(
            "liquidity",
            document_ids=[9],
            limit=5,
            query_plan=plan,
            log_stage="unit_sparse",
        )
    finally:
        logger.remove(handler_id)
    assert results == []
    assert fake_client.search_calls[0]["anns_field"] == "sparse"
    # The OpenSearch field-boost plan cannot map onto Milvus's single BM25
    # field; the stage log must say so (doc §3.9 mitigation).
    assert any("query_plan_applied" in line and '"unit_sparse"' in line for line in logged)


def test_sparse_backend_search_delegates_to_store(fake_client: FakeClient) -> None:
    fake_client.search_result = [
        [
            {
                "id": "uuid-s2",
                "distance": 3.25,
                "entity": {
                    "document_id": 9,
                    "parent_id": None,
                    "node_type": "chunk",
                    "level": 0,
                    "order_index": 1,
                    "title": "T",
                    "metadata": {},
                    "retrieval_fields": {},
                    "text": "abc",
                },
            }
        ]
    ]
    out = asyncio.run(
        MilvusSparseBackend().search([9], "abc", limit=3, log_stage="unit_backend")
    )
    assert out[0]["node_id"] == "uuid-s2"
    assert out[0]["sparse_score"] == 3.25


def test_sparse_backend_replace_document_nodes_is_noop(fake_client: FakeClient) -> None:
    assert asyncio.run(MilvusSparseBackend().replace_document_nodes(9, [])) is None
    # No Milvus interaction: rows are owned by the dense backend's replace path.
    assert fake_client.search_calls == []
    assert fake_client.inserted == []
    assert fake_client.deleted_filters == []


def test_factory_sparse_milvus_requires_milvus_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    factory.get_sparse_backend.cache_clear()
    monkeypatch.setattr(config, "sparse_backend", "milvus")
    monkeypatch.setattr(config, "dense_backend", "qdrant")
    with pytest.raises(ValueError, match="SPARSE_BACKEND=milvus requires DENSE_BACKEND=milvus"):
        factory.get_sparse_backend()

    factory.get_sparse_backend.cache_clear()
    monkeypatch.setattr(config, "dense_backend", "milvus")
    assert isinstance(factory.get_sparse_backend(), MilvusSparseBackend)
    factory.get_sparse_backend.cache_clear()


# ---------- BM25 text enrichment (title/search_hints prefix) ----------

def test_build_enriched_text_repeats_and_prefix() -> None:
    text, prefix = milvus_store.build_enriched_text(
        "Net Sales",
        "net sales revenue md&a",
        "Body paragraph.",
        title_repeats=2,
        hints_repeats=3,
    )
    assert text == "Net Sales\nNet Sales\n" + "net sales revenue md&a\n" * 2 + "net sales revenue md&a\nBody paragraph."
    assert text[prefix:] == "Body paragraph."
    assert text[:prefix].count("Net Sales") == 2
    assert text[:prefix].count("net sales revenue md&a") == 3

def test_build_enriched_text_disabled_returns_pure_body() -> None:
    text, prefix = milvus_store.build_enriched_text(
        "T", "hints", "Body.", title_repeats=0, hints_repeats=0
    )
    assert (text, prefix) == ("Body.", 0)
    # Empty title/hints behave like disabled even with repeats configured.
    text2, prefix2 = milvus_store.build_enriched_text("", "", "Body.", title_repeats=2, hints_repeats=3)
    assert (text2, prefix2) == ("Body.", 0)

def test_build_enriched_text_truncation_keeps_prefix() -> None:
    text, prefix = milvus_store.build_enriched_text(
        "T", "h", "x" * 70_000, title_repeats=2, hints_repeats=3
    )
    assert len(text) == milvus_store._MAX_TEXT_CHARS
    assert text.startswith("T\nT\nh\nh\nh\n")
    assert text[prefix:] == "x" * (milvus_store._MAX_TEXT_CHARS - prefix)


def test_node_to_row_enriches_text_and_records_prefix(
    fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "milvus_text_title_repeats", 2)
    monkeypatch.setattr(config, "milvus_text_hints_repeats", 3)
    row = milvus_store._node_to_row(_node())
    prefix = row["metadata"][milvus_store._TEXT_PREFIX_META_KEY]
    assert row["text"].startswith("Business\nBusiness\n")
    assert row["text"][prefix:] == "Apple Inc. revenue discussion."

def test_search_hits_strip_enrichment_prefix(fake_client: FakeClient) -> None:
    prefix_meta = {milvus_store._TEXT_PREFIX_META_KEY: 17}  # len("TITTLEPREFIX12345")
    fake_client.search_result = [
        [
            {
                "id": "uuid-e",
                "distance": 1.5,
                "entity": {
                    "document_id": 9,
                    "parent_id": None,
                    "node_type": "chunk",
                    "level": 0,
                    "order_index": 1,
                    "title": "T",
                    "metadata": prefix_meta,
                    "text": "TITTLEPREFIX12345Body text here.",
                },
            }
        ]
    ]
    dense = milvus_store.dense_search([0.1, 0.2], document_ids=[9], limit=3)
    assert dense[0]["text_preview"] == "Body text here."
    sparse = milvus_store.sparse_search("body", document_ids=[9], limit=3)
    assert sparse[0]["text"] == "Body text here."

def test_rebuild_texts_roundtrip_idempotent(
    fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "milvus_text_title_repeats", 2)
    monkeypatch.setattr(config, "milvus_text_hints_repeats", 3)
    fake_client.probe_result = [{"document_id": 9}]
    fake_client.query_rows = [
        {
            "id": "uuid-r1",
            "document_id": 9,
            "parent_id": None,
            "node_type": "chunk",
            "level": 0,
            "order_index": 1,
            "title": "Net Sales",
            "metadata": {"_retrieval_fields": {"search_hints": "net sales"}},
            "retrieval_fields": {"domain": ["finance"]},
            "text": "Original body.",  # legacy row: no prefix marker
            "dense": [0.1, 0.2],
        },
        {
            "id": "uuid-r2",
            "document_id": 9,
            "parent_id": None,
            "node_type": "chunk",
            "level": 0,
            "order_index": 2,
            "title": "Europe",
            "metadata": {
                "_retrieval_fields": {"search_hints": "europe segment"},
                milvus_store._TEXT_PREFIX_META_KEY: 7 + 1,
            },
            "retrieval_fields": {},
            "text": "Europe\nEurope\nbody two.",  # already enriched: prefix stripped before re-build
            "dense": [0.3, 0.4],
        },
    ]
    stats = milvus_store.rebuild_texts()
    assert stats["rows_seen"] == 2
    assert stats["rows_changed"] == 2
    assert stats["rows_upserted"] == 2
    by_id = {row["id"]: row for row in fake_client.upserted}
    assert by_id["uuid-r1"]["text"].startswith("Net Sales\nNet Sales\nnet sales\nnet sales\nnet sales\n")
    assert by_id["uuid-r1"]["dense"] == [0.1, 0.2]  # dense passes through untouched
    assert by_id["uuid-r2"]["text"].endswith("\nbody two.")
    assert by_id["uuid-r2"]["text"].count("Europe") == 2  # old prefix stripped, rebuilt once per repeats

    # Idempotent: feed upserted rows back as the new state; nothing changes.
    fake_client.query_rows = [dict(row) for row in fake_client.upserted]
    fake_client.upserted.clear()
    stats2 = milvus_store.rebuild_texts()
    assert stats2["rows_changed"] == 0
    assert fake_client.upserted == []
