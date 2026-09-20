"""R3 embedding-based RAGAS metrics routing in tools/evaluation_pipeline.py.

Fully offline: the ragas metric classes and the LLM factory are replaced with
fakes recording constructor kwargs, and the embeddings client builder is
monkeypatched. Only routing / degradation behaviour is asserted.
"""

from __future__ import annotations

import asyncio

import tools.evaluation_pipeline as ep
from core.config import config


def _job(reference: str | None = "iPhone revenue was $51.3B in FY24.") -> dict:
    return {
        "id": "job-1",
        "trace_id": "trace-1",
        "query": "What was iPhone revenue?",
        "answer": "iPhone revenue was $51.3B.",
        "context_json": [{"text": "iPhone revenue context"}],
        "reference": reference,
    }


def _make_fake_metric(constructed: dict, name: str, score: float = 0.5):
    class FakeMetric:
        def __init__(self, **kwargs):
            constructed[name] = kwargs

        async def single_turn_ascore(self, sample):
            return score

    return FakeMetric


def _patch_metrics(monkeypatch, constructed: dict) -> None:
    """Replace every ragas metric class + the LLM factory with offline fakes."""
    monkeypatch.setattr(ep, "get_llm", lambda *a, **kw: object())
    for name in (
        "Faithfulness",
        "LLMContextPrecisionWithoutReference",
        "LLMContextPrecisionWithReference",
        "LLMContextRecall",
        "FactualCorrectness",
        "AnswerCorrectness",
        "AnswerSimilarity",
        "ResponseRelevancy",
    ):
        monkeypatch.setattr(ep, name, _make_fake_metric(constructed, name))


_EMBED_SENTINEL = object()

_REFERENCE_METRICS = {"context_precision_with_reference", "context_recall", "factual_correctness"}
_EMBEDDING_METRICS = {"answer_correctness", "answer_similarity", "answer_relevancy"}
# _make_fake_metric records constructor kwargs under the patched class attribute name.
_EMBEDDING_METRIC_CLASSES = {"AnswerCorrectness", "AnswerSimilarity", "ResponseRelevancy"}


def test_flag_off_reference_job_scores_four_metrics(monkeypatch):
    constructed: dict = {}
    _patch_metrics(monkeypatch, constructed)
    monkeypatch.setattr(config, "eval_embeddings_metrics_enabled", False)

    scores = asyncio.run(ep._score_job(_job()))

    assert set(scores) == {"faithfulness"} | _REFERENCE_METRICS
    assert not _EMBEDDING_METRIC_CLASSES & set(constructed)  # never constructed


def test_flag_on_appends_three_embedding_metrics(monkeypatch):
    constructed: dict = {}
    _patch_metrics(monkeypatch, constructed)
    monkeypatch.setattr(config, "eval_embeddings_metrics_enabled", True)
    monkeypatch.setattr(ep, "_build_ragas_embeddings", lambda: _EMBED_SENTINEL)

    scores = asyncio.run(ep._score_job(_job()))

    assert set(scores) == {"faithfulness"} | _REFERENCE_METRICS | _EMBEDDING_METRICS
    # Embeddings client wired into every embedding-based metric constructor.
    assert constructed["AnswerCorrectness"]["embeddings"] is _EMBED_SENTINEL
    assert constructed["AnswerCorrectness"]["llm"] is not None
    assert constructed["AnswerSimilarity"]["embeddings"] is _EMBED_SENTINEL
    assert "llm" not in constructed["AnswerSimilarity"]  # similarity is embeddings-only
    assert constructed["ResponseRelevancy"]["embeddings"] is _EMBED_SENTINEL


def test_flag_on_embeddings_build_failure_degrades_without_raising(monkeypatch):
    constructed: dict = {}
    _patch_metrics(monkeypatch, constructed)
    monkeypatch.setattr(config, "eval_embeddings_metrics_enabled", True)
    monkeypatch.setattr(ep, "_build_ragas_embeddings", lambda: None)  # no key / unsupported

    scores = asyncio.run(ep._score_job(_job()))

    assert set(scores) == {"faithfulness"} | _REFERENCE_METRICS
    assert not _EMBEDDING_METRIC_CLASSES & set(constructed)


def test_flag_on_missing_metric_classes_degrades_without_raising(monkeypatch):
    constructed: dict = {}
    _patch_metrics(monkeypatch, constructed)
    monkeypatch.setattr(config, "eval_embeddings_metrics_enabled", True)
    monkeypatch.setattr(ep, "_build_ragas_embeddings", lambda: _EMBED_SENTINEL)
    # Simulate ragas build without the embedding metrics (import fallback path).
    monkeypatch.setattr(ep, "AnswerSimilarity", None)

    scores = asyncio.run(ep._score_job(_job()))

    assert set(scores) == {"faithfulness"} | _REFERENCE_METRICS


def test_reference_free_route_ignores_embeddings_flag(monkeypatch):
    constructed: dict = {}
    _patch_metrics(monkeypatch, constructed)
    monkeypatch.setattr(config, "eval_embeddings_metrics_enabled", True)
    monkeypatch.setattr(
        ep, "_build_ragas_embeddings", lambda: (_ for _ in ()).throw(AssertionError("must not build"))
    )

    scores = asyncio.run(ep._score_job(_job(reference=None)))

    assert set(scores) == {"faithfulness", "context_precision"}


def test_build_ragas_embeddings_no_provider_returns_none(monkeypatch):
    from tools import vectorizer

    monkeypatch.setattr(vectorizer, "get_available_api", lambda: None)

    assert ep._build_ragas_embeddings() is None


def test_build_ragas_embeddings_construction_failure_returns_none(monkeypatch):
    import langchain_openai
    from tools import vectorizer

    monkeypatch.setattr(vectorizer, "get_available_api", lambda: "qwen")

    class ExplodingEmbeddings:
        def __init__(self, **kwargs):
            raise RuntimeError("no credentials")

    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", ExplodingEmbeddings)

    assert ep._build_ragas_embeddings() is None


def test_build_ragas_embeddings_positive_qwen(monkeypatch):
    from tools import vectorizer

    monkeypatch.setattr(vectorizer, "get_available_api", lambda: "qwen")
    monkeypatch.setattr(config, "qwen_api_key", "test-key")
    monkeypatch.setattr(config, "embedding_model", "text-embedding-v3")

    emb = ep._build_ragas_embeddings()

    assert emb is not None
    assert type(emb).__name__ == "LangchainEmbeddingsWrapper"
