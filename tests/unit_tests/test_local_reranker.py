"""Unit tests for the local CrossEncoder reranker and the composite/factory selection."""

from __future__ import annotations

import asyncio

import pytest
from core.config import config
from tools import rerank as rerank_module
from tools.local_reranker import LocalReranker
from tools.rerank import CompositeReranker, TruncateReranker, get_reranker


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
    # text_preview is used when text is missing (Bocha text extraction parity).
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


def _fake_backend(marker: str):
    class FakeBackend:
        def __init__(self) -> None:
            self.called = False

        def describe_config(self):
            return {"backend": marker}

        async def rerank(self, *, query, candidates, top_n, out_stats=None):
            self.called = True
            stats = out_stats if out_stats is not None else {}
            stats.clear()
            out = [dict(row, rerank_score=0.42) for row in candidates[:top_n]]
            stats.update({"mode": "fake_success", "candidates_out": len(out)})
            return out

    return FakeBackend()


def test_composite_escalates_when_local_model_unavailable():
    local = LocalReranker()
    local._load_error = "RuntimeError: no model"  # simulate a failed lazy load
    fallback = _fake_backend("bocha-fake")
    composite = CompositeReranker([local, fallback])

    stats: dict = {}
    out = asyncio.run(
        composite.rerank(query="q", candidates=_candidates(), top_n=3, out_stats=stats)
    )
    assert fallback.called is True
    assert all(row["rerank_score"] == 0.42 for row in out)
    assert stats["mode"] == "local_model_unavailable"
    assert stats["escalated_to"] == "FakeBackend"
    assert stats["fallback_stage"]["mode"] == "fake_success"


def test_composite_prefers_local_when_available():
    local = LocalReranker(model=FakeCrossEncoder({"alpha passage": 1.0}))
    fallback = _fake_backend("bocha-fake")
    composite = CompositeReranker([local, fallback])

    out = asyncio.run(composite.rerank(query="q", candidates=_candidates(), top_n=3))
    assert fallback.called is False
    assert out[0]["node_id"] == "a"


def test_truncate_reranker_disabled_mode():
    reranker = TruncateReranker()
    stats: dict = {}
    out = asyncio.run(
        reranker.rerank(query="q", candidates=_candidates(), top_n=2, out_stats=stats)
    )
    assert [row["node_id"] for row in out] == ["a", "b"]
    assert stats["mode"] == "skipped_disabled"


@pytest.mark.parametrize(
    ("backend", "first_type", "chain_len"),
    [
        ("bocha", rerank_module.BochaReranker, 1),
        ("none", TruncateReranker, 1),
    ],
)
def test_factory_backend_selection(backend: str, first_type: type, chain_len: int, monkeypatch):
    get_reranker.cache_clear()
    monkeypatch.setattr(config, "reranker_backend", backend)
    composite = get_reranker()
    assert len(composite._chain) == chain_len
    assert isinstance(composite._chain[0], first_type)
    get_reranker.cache_clear()


def test_warmup_loads_local_when_primary(monkeypatch):
    import threading

    loaded = threading.Event()

    class SpyLocal(LocalReranker):
        def ensure_loaded(self) -> bool:
            loaded.set()
            return True

    spy = SpyLocal(model=None)
    monkeypatch.setattr(
        rerank_module, "get_reranker", lambda: CompositeReranker([spy])
    )
    asyncio.run(rerank_module.warmup_reranker())
    assert loaded.is_set() is True


def test_warmup_noop_when_local_not_primary(monkeypatch):
    import threading

    loaded = threading.Event()

    class SpyLocal(LocalReranker):
        def ensure_loaded(self) -> bool:
            loaded.set()
            return True

    # backend=bocha -> chain[0] is BochaReranker, warmup must not touch local.
    monkeypatch.setattr(config, "reranker_backend", "bocha")
    rerank_module.get_reranker.cache_clear()
    asyncio.run(rerank_module.warmup_reranker())
    assert loaded.is_set() is False
    rerank_module.get_reranker.cache_clear()


def test_factory_local_default_includes_bocha_fallback(monkeypatch):
    get_reranker.cache_clear()
    monkeypatch.setattr(config, "reranker_backend", "local")
    monkeypatch.setattr(config, "bocha_reranker_url", "https://api.bocha.cn/v1/rerank")
    monkeypatch.setattr(config, "bocha_api_key", "sk-test")
    composite = get_reranker()
    assert isinstance(composite._chain[0], LocalReranker)
    assert isinstance(composite._chain[1], rerank_module.BochaReranker)
    described = composite.describe_config()
    assert described["backend"] == "local" and len(described["chain"]) == 2
    get_reranker.cache_clear()
