"""Unit tests for the local CrossEncoder reranker and the factory selection."""

from __future__ import annotations

import asyncio

import pytest
from core.config import config
from tools import rerank as rerank_module
from tools.local_reranker import LocalReranker
from tools.rerank import TruncateReranker, get_reranker


class FakeCrossEncoder:
    """Mimics sentence_transformers.CrossEncoder.predict over (query, doc) pairs."""

    def __init__(self, scores: dict[str, float], *, error: Exception | None = None) -> None:
        self._scores = scores
        self._error = error
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, batch_size=8, show_progress_bar=False, activation_fn=None):
        if self._error is not None:
            raise self._error
        self.calls.append(list(pairs))
        return [self._scores.get(doc, 0.0) for _, doc in pairs]


def _candidates() -> list[dict]:
    return [
        {"node_id": "a", "text": "alpha passage"},
        {"node_id": "b", "text_preview": "beta preview"},
        {"node_id": "c", "text": "gamma passage"},
    ]


def test_local_rerank_orders_by_score_and_respects_top_n():
    fake = FakeCrossEncoder({"alpha passage": 0.2, "beta preview": 0.9, "gamma passage": 0.5})
    reranker = LocalReranker(model=fake)
    stats: dict = {}
    out = asyncio.run(
        reranker.rerank(query="q", candidates=_candidates(), top_n=2, out_stats=stats)
    )

    assert [row["node_id"] for row in out] == ["b", "c"]
    assert out[0]["rerank_score"] == pytest.approx(0.9)
    assert stats["mode"] == "local_success"
    assert stats["candidates_in"] == 3 and stats["candidates_out"] == 2
    # text_preview is used when text is missing.
    assert ("q", "beta preview") in fake.calls[0]


def test_local_rerank_empty_candidates_returns_empty():
    reranker = LocalReranker(model=FakeCrossEncoder({}))
    stats: dict = {}
    out = asyncio.run(reranker.rerank(query="q", candidates=[], top_n=5, out_stats=stats))
    assert out == [] and stats["mode"] == "skipped_empty_candidates"


def test_local_rerank_predict_error_truncates_fusion_order():
    fake = FakeCrossEncoder({}, error=RuntimeError("boom"))
    reranker = LocalReranker(model=fake)
    stats: dict = {}
    out = asyncio.run(
        reranker.rerank(query="q", candidates=_candidates(), top_n=2, out_stats=stats)
    )
    assert [row["node_id"] for row in out] == ["a", "b"]
    assert stats["mode"] == "local_error_fallback"
    assert stats["fallback"] == "truncate_fusion_order"


def test_local_model_unavailable_truncates_fusion_order():
    local = LocalReranker()
    local._load_error = "RuntimeError: no model"  # simulate a failed lazy load

    stats: dict = {}
    out = asyncio.run(local.rerank(query="q", candidates=_candidates(), top_n=2, out_stats=stats))
    assert [row["node_id"] for row in out] == ["a", "b"]
    assert stats["mode"] == "local_model_unavailable"
    assert stats["fallback"] == "truncate_fusion_order"
    assert stats["error"] == "RuntimeError: no model"


def test_truncate_reranker_disabled_mode():
    reranker = TruncateReranker()
    stats: dict = {}
    out = asyncio.run(
        reranker.rerank(query="q", candidates=_candidates(), top_n=2, out_stats=stats)
    )
    assert [row["node_id"] for row in out] == ["a", "b"]
    assert stats["mode"] == "skipped_disabled"


@pytest.mark.parametrize(
    ("backend", "expected_type"),
    [
        ("local", LocalReranker),
        ("none", TruncateReranker),
    ],
)
def test_factory_backend_selection(backend: str, expected_type: type, monkeypatch):
    get_reranker.cache_clear()
    monkeypatch.setattr(config, "reranker_backend", backend)
    assert isinstance(get_reranker(), expected_type)
    get_reranker.cache_clear()


def test_warmup_loads_local_when_primary(monkeypatch):
    import threading

    loaded = threading.Event()

    class SpyLocal(LocalReranker):
        def ensure_loaded(self) -> bool:
            loaded.set()
            return True

    spy = SpyLocal(model=None)
    monkeypatch.setattr(rerank_module, "get_reranker", lambda: spy)
    asyncio.run(rerank_module.warmup_reranker())
    assert loaded.is_set() is True


def test_warmup_noop_when_local_not_primary(monkeypatch):
    import threading

    loaded = threading.Event()

    class SpyLocal(LocalReranker):
        def ensure_loaded(self) -> bool:
            loaded.set()
            return True

    # backend=none -> TruncateReranker, warmup must not touch the local model.
    monkeypatch.setattr(config, "reranker_backend", "none")
    rerank_module.get_reranker.cache_clear()
    asyncio.run(rerank_module.warmup_reranker())
    assert loaded.is_set() is False
    rerank_module.get_reranker.cache_clear()


def test_factory_local_default(monkeypatch):
    get_reranker.cache_clear()
    monkeypatch.setattr(config, "reranker_backend", "local")
    reranker = get_reranker()
    assert isinstance(reranker, LocalReranker)
    described = reranker.describe_config()
    assert described["backend"] == "local"
    get_reranker.cache_clear()
