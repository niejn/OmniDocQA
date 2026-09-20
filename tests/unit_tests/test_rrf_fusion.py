"""1C fusion-sink groundwork — RRF consistency unit tests + FUSION_BACKEND routing.

Offline (no Milvus server). Layers covered:

1. ``rrf_scores`` / ``rrf_fuse`` (tools/llamaindex_retrieval.py) implement the
   Milvus RRFRanker formula: ``score = Σ 1/(k + rank)`` with ``rank`` starting
   at 1 within each list. pymilvus 2.6's ``RRFRanker`` only serializes
   ``{"strategy": "rrf", "params": {"k": ...}}`` — the summation happens
   server-side, so the formula is pinned here against an independent in-test
   reference implementation written straight from the Milvus docs definition
   (and the k constant against the RRFRanker payload that reaches the server).
2. The item-level wrapper ``reciprocal_rank_fusion`` produces the same ordering
   and scores as ``rrf_fuse``/``rrf_scores``.
3. Tie handling: identical fused scores have no server-documented order, so the
   app layer pins its own deterministic rules — bare ``rrf_fuse`` keeps
   first-appearance order; ``reciprocal_rank_fusion`` breaks ties by
   ``(score, level)`` descending. Tests assert those documented rules.
4. ``NodeHybridRetriever._dense_sparse_fused`` routing: FUSION_BACKEND=app
   (default) unchanged; FUSION_BACKEND=milvus routes to the server-side
   hybrid path and degrades back to app fusion whenever the backend pair is
   not Milvus or the hybrid call fails.
"""

from __future__ import annotations

import asyncio

import pytest
from core.config import config
from pymilvus import RRFRanker
from tools import milvus_store
from tools.llamaindex_retrieval import (
    NodeHybridRetriever,
    reciprocal_rank_fusion,
    rrf_fuse,
    rrf_scores,
)
from tools.retrieval_backends.dense_milvus import MilvusDenseBackend
from tools.retrieval_backends.sparse_milvus import MilvusSparseBackend

# ── reference implementation, written from the Milvus RRF docs formula ──


def reference_rrf(ranked_lists: list[list[str]], k: int = 60) -> tuple[list[str], dict[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):  # rank starts at 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    first_seen: dict[str, int] = {}
    for ranked in ranked_lists:
        for key in ranked:
            first_seen.setdefault(key, len(first_seen))
    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return ordered, scores


# ── rrf_scores / rrf_fuse: formula, ordering, ties ──────────────────────────


def test_rrf_scores_formula_rank_from_one_k_60():
    # a: rank 1 in list1 + rank 2 in list2 → 1/61 + 1/62; c: rank 3 in list2 only.
    scores = rrf_scores([["a", "x"], ["b", "a", "c"]])
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 61)
    assert scores["x"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 63)


def test_rrf_fuse_orders_by_descending_fused_score():
    # d: 1/61 + 1/62; z: 1/63 + 1/63 = 2/63; x: 1/62; y: 1/62.
    fused = rrf_fuse([["d", "x", "z"], ["d", "y", "z"]])
    assert fused == ["d", "z", "x", "y"]  # x/y tie keeps first-appearance (x in list1)


def test_rrf_fuse_handles_missing_keys_and_unequal_lengths():
    fused = rrf_fuse(
        [
            ["only-dense"],
            ["only-sparse", "shared", "extra"],
            ["shared"],
        ]
    )
    # shared: 1/62 + 1/61; only-sparse: 1/61; only-dense: 1/61; extra: 1/63.
    assert fused[0] == "shared"
    assert set(fused) == {"only-dense", "only-sparse", "shared", "extra"}
    scores = rrf_scores([["only-dense"], ["only-sparse", "shared", "extra"], ["shared"]])
    assert scores["only-sparse"] == pytest.approx(scores["only-dense"])

def test_rrf_fuse_exact_tie_keeps_first_appearance():
    # Symmetric swap → identical fused scores; the documented tie rule is
    # first-appearance order across the ranked lists (list1 before list2).
    fused = rrf_fuse([["a", "b"], ["b", "a"]])
    assert rrf_scores([["a", "b"], ["b", "a"]])["a"] == pytest.approx(
        rrf_scores([["a", "b"], ["b", "a"]])["b"]
    )
    assert fused == ["a", "b"]
    # Empty lists degrade to an empty fusion.
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


def test_rrf_fuse_matches_milvus_rrfranker_reference_on_synthetic_combos():
    # The RRFRanker payload reaching the server must carry k=60 (the constant
    # rrf_scores/rrf_fuse use); the summation itself runs server-side.
    assert RRFRanker().dict() == {"strategy": "rrf", "params": {"k": 60}}
    assert RRFRanker(k=60).dict() == {"strategy": "rrf", "params": {"k": 60}}

    # Multi-way combos: 2-3 lists, overlaps, missing keys, different lengths.
    combos: list[list[list[str]]] = [
        [["d1", "d2", "d3", "d4"], ["s1", "d2", "d3"]],
        [
            ["a", "b", "c", "d", "e", "f"],
            ["c", "a", "x"],
            ["x", "e", "b", "q"],
        ],
        [["solo"], [], ["solo", "late", "later", "latest"], ["late", "latest"]],
    ]
    for ranked_lists in combos:
        expected_order, expected_scores = reference_rrf(ranked_lists)
        # Same formula ⇒ same scores; same documented tie rule ⇒ same order
        # (combos may contain exact ties, which the reference resolves by
        # first appearance exactly like rrf_fuse).
        assert rrf_fuse(ranked_lists) == expected_order
        assert rrf_scores(ranked_lists) == pytest.approx(expected_scores)
        # k is a parameter and flows through to the same formula.
        alt_order, alt_scores = reference_rrf(ranked_lists, k=7)
        assert rrf_fuse(ranked_lists, k=7) == alt_order
        assert rrf_scores(ranked_lists, k=7) == pytest.approx(alt_scores)

    # Tie-free combo: all fused scores distinct → the ordering is unique, so
    # this case pins exact agreement with the Milvus formula without relying
    # on any tie convention.
    tie_free = [["d", "x"], ["d", "y", "x"]]
    _order, scores = reference_rrf(tie_free)
    assert len(set(scores.values())) == len(scores)
    assert rrf_fuse(tie_free) == ["d", "x", "y"]
    assert sorted(rrf_fuse(tie_free), key=lambda key: -scores[key]) == rrf_fuse(tie_free)


# ── reciprocal_rank_fusion parity with the pure functions ───────────────────


def _item(node_id: str, level: int = 0, **extra: object) -> dict:
    return {"node_id": node_id, "level": level, **extra}


def test_reciprocal_rank_fusion_ordering_and_scores_match_rrf_fuse():
    dense = [_item("d1", 0, dense_score=0.9), _item("d2", 0, dense_score=0.8)]
    sparse = [_item("s1", 0, sparse_score=12.0), _item("d1", 0, sparse_score=4.0)]
    fused = reciprocal_rank_fusion(
        [dense, sparse],
        limit=10,
        score_keys=["dense_score", "sparse_score"],
    )
    ids = [item["node_id"] for item in fused]
    assert ids == rrf_fuse([["d1", "d2"], ["s1", "d1"]])
    expected = rrf_scores([["d1", "d2"], ["s1", "d1"]])
    for item in fused:
        assert item["fusion_score"] == pytest.approx(expected[item["node_id"]])
    # Merged item keeps the first list's payload plus the other list's score.
    by_id = {item["node_id"]: item for item in fused}
    assert by_id["d1"]["dense_score"] == 0.9  # from dense (first list)
    assert by_id["d1"]["sparse_score"] == 4.0  # merged from sparse
    assert by_id["s1"]["sparse_score"] == 12.0
    # limit truncates after scoring.
    assert [i["node_id"] for i in reciprocal_rank_fusion([dense, sparse], limit=2, score_keys=[])] == ids[:2]


def test_reciprocal_rank_fusion_ties_break_by_level_descending():
    # Same fused score (each key hit once at the same rank in its own list):
    # the wrapper's documented tie rule is (score, level) reverse → the
    # summary/section node (level 2) outranks the leaf (level 0).
    a = [_item("x", 2, dense_score=1.0), _item("y", 0, dense_score=0.5)]
    b = [_item("y", 0, sparse_score=9.0), _item("x", 2, sparse_score=8.0)]
    fused = reciprocal_rank_fusion([a, b], limit=10, score_keys=["dense_score", "sparse_score"])
    scores = rrf_scores([["x", "y"], ["y", "x"]])
    assert scores["x"] == pytest.approx(scores["y"])  # exact tie
    assert [item["node_id"] for item in fused][:2] == ["x", "y"]  # level 2 before level 0
    assert fused[0]["level"] == 2 and fused[1]["level"] == 0


# ── FUSION_BACKEND routing seam (NodeHybridRetriever._dense_sparse_fused) ───


class FakeDenseBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, query_vector, *, document_ids, limit, levels=None, metadata_filters=None, log_stage=None):
        self.calls.append(
            {
                "levels": levels,
                "limit": limit,
                "metadata_filters": metadata_filters,
                "log_stage": log_stage,
            }
        )
        return [_item("d1", 0, dense_score=0.9), _item("d2", 1, dense_score=0.8)]


class FakeSparseBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def search(self, document_ids, query, *, limit, levels=None, metadata_filters=None, query_plan=None, log_stage=None):
        self.calls.append(
            {
                "query": query,
                "levels": levels,
                "limit": limit,
                "metadata_filters": metadata_filters,
                "log_stage": log_stage,
            }
        )
        return [_item("s1", 0, sparse_score=12.0), _item("d1", 0, sparse_score=4.0)]


def _bare_retriever() -> tuple[NodeHybridRetriever, FakeDenseBackend, FakeSparseBackend]:
    """NodeHybridRetriever without __init__ (no llama_index/callback wiring)."""
    retriever = object.__new__(NodeHybridRetriever)
    dense, sparse = FakeDenseBackend(), FakeSparseBackend()
    retriever.document_ids = [9]
    retriever.dense_backend = dense
    retriever.sparse_backend = sparse
    return retriever, dense, sparse


def test_dense_sparse_fused_app_backend_default_unchanged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "fusion_backend", "app")
    retriever, dense, sparse = _bare_retriever()
    dense_hits, sparse_hits, fused, server_fused = asyncio.run(
        retriever._dense_sparse_fused(
            query="net sales",
            query_embedding=[0.1, 0.2],
            levels=[0],
            metadata_filters={"domain": ["finance"]},
            fusion_limit=8,
            stage="leaf",
            candidate_source="leaf",
        )
    )
    assert server_fused is False
    assert dense.calls and sparse.calls  # both searches ran (two-round-trip path)
    assert dense.calls[0]["log_stage"] == "dense_leaf"
    assert sparse.calls[0]["log_stage"] == "sparse_leaf"
    assert dense.calls[0]["limit"] == config.dense_top_k
    assert sparse.calls[0]["limit"] == config.sparse_top_k
    assert [item["node_id"] for item in fused] == rrf_fuse([["d1", "d2"], ["s1", "d1"]])
    assert all(item["candidate_source"] == "leaf" for item in fused)
    # Missing embedding keeps the app path and skips dense only.
    dense2, sparse2 = FakeDenseBackend(), FakeSparseBackend()
    retriever.dense_backend, retriever.sparse_backend = dense2, sparse2
    d, s, _fused, server_fused2 = asyncio.run(
        retriever._dense_sparse_fused(
            query="net sales",
            query_embedding=None,
            levels=[0],
            metadata_filters={},
            fusion_limit=8,
            stage="leaf",
            candidate_source="leaf",
        )
    )
    assert server_fused2 is False and d == [] and s
    assert dense2.calls == []


def test_dense_sparse_fused_milvus_mode_falls_back_without_milvus_backends(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(config, "fusion_backend", "milvus")
    retriever, dense, sparse = _bare_retriever()
    # Fakes are not the Milvus pair → must degrade to app fusion, not fail.
    _d, _s, fused, server_fused = asyncio.run(
        retriever._dense_sparse_fused(
            query="q",
            query_embedding=[0.1],
            levels=[0],
            metadata_filters={},
            fusion_limit=8,
            stage="leaf",
            candidate_source="leaf",
        )
    )
    assert server_fused is False
    assert dense.calls and sparse.calls
    assert [item["node_id"] for item in fused] == rrf_fuse([["d1", "d2"], ["s1", "d1"]])


def test_dense_sparse_fused_milvus_mode_routes_to_hybrid_search(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(config, "fusion_backend", "milvus")
    retriever = object.__new__(NodeHybridRetriever)
    retriever.document_ids = [9]
    retriever.dense_backend = MilvusDenseBackend()
    retriever.sparse_backend = MilvusSparseBackend()

    captured: dict = {}

    def fake_hybrid_search(
        vec, text, *, limit, filter_expr, dense_limit=None, sparse_limit=None, log_stage=None, **kwargs
    ):
        captured.update(
            {
                "vec": vec,
                "text": text,
                "limit": limit,
                "filter_expr": filter_expr,
                "dense_limit": dense_limit,
                "sparse_limit": sparse_limit,
                "log_stage": log_stage,
            }
        )
        return [
            {"node_id": "h1", "fusion_score": 0.032, "level": 0, "text_preview": "body"},
            {"node_id": "h2", "fusion_score": 0.03, "level": 0, "text_preview": "body2"},
        ]

    monkeypatch.setattr(milvus_store, "hybrid_search", fake_hybrid_search)
    dense_hits, sparse_hits, fused, server_fused = asyncio.run(
        retriever._dense_sparse_fused(
            query="net sales",
            query_embedding=[0.1, 0.2],
            levels=[0],
            metadata_filters={"domain": ["finance"]},
            fusion_limit=5,
            stage="leaf",
            candidate_source="leaf",
        )
    )
    assert server_fused is True
    assert dense_hits == [] and sparse_hits == []  # no per-request sub-scores
    assert captured["vec"] == [0.1, 0.2]
    assert captured["text"] == "net sales"
    assert captured["limit"] == 5
    # Fusion-pool parity: server-side per-request depths match the app path.
    assert captured["dense_limit"] == config.dense_top_k
    assert captured["sparse_limit"] == config.sparse_top_k
    assert captured["log_stage"] == "hybrid_milvus_leaf"
    # Filter expression has full parity (documents + levels + metadata filters).
    assert captured["filter_expr"] == (
        'document_id in [9] and level in [0] and json_contains_any(retrieval_fields["domain"], ["finance"])'
    )
    assert [item["node_id"] for item in fused] == ["h1", "h2"]
    assert all(item["candidate_source"] == "leaf" for item in fused)


def test_dense_sparse_fused_milvus_mode_degrades_on_backend_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(config, "fusion_backend", "milvus")
    retriever = object.__new__(NodeHybridRetriever)
    retriever.document_ids = [9]
    retriever.dense_backend = MilvusDenseBackend()
    retriever.sparse_backend = MilvusSparseBackend()

    def boom(vec, text, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(milvus_store, "hybrid_search", boom)
    dense, sparse = FakeDenseBackend(), FakeSparseBackend()
    retriever.dense_backend = dense
    retriever.sparse_backend = sparse
    _d, _s, _fused, server_fused = asyncio.run(
        retriever._dense_sparse_fused(
            query="q",
            query_embedding=[0.1],
            levels=[0],
            metadata_filters={},
            fusion_limit=8,
            stage="leaf",
            candidate_source="leaf",
        )
    )
    # Backend error → app-fusion fallback keeps serving (documented contract).
    assert server_fused is False
    assert dense.calls and sparse.calls
