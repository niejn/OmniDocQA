"""Async RAGAS evaluation worker and Langfuse score writeback."""

from __future__ import annotations

import json
from typing import Any

from core.config import config
from loguru import logger

from .langfuse_tracing import tracer
from .llm import get_llm
from .node_repository import complete_evaluation_job, list_pending_evaluation_jobs

try:
    # Windows + Py3.13: pyarrow's native extension raises an access violation when
    # first imported lazily inside a running event loop (via ragas -> datasets).
    # Warm it at module import time, before any asyncio code runs.
    import pyarrow  # noqa: F401
except ImportError:  # pragma: no cover
    pass

try:
    # ragas<=0.4.3 + langchain-community>=0.4: stub the removed vertexai module
    # before the first ragas import or everything below degrades to None.
    from .ragas_compat import ensure_ragas_importable

    ensure_ragas_importable()
    from ragas.dataset_schema import SingleTurnSample

    # NOTE: keep the legacy `ragas.metrics` path — the new `ragas.metrics.collections`
    # classes require InstructorLLM (llm_factory) and reject our langchain get_llm.
    # Migrate to collections + llm_factory once in R3 (doc §6.3) together with the
    # renamed class (LLMContextPrecisionWithoutReference -> ContextPrecisionWithoutReference).
    # R2 (doc §6.2): reference-based route. All three are LLM-only (no
    # embeddings dependency — AnswerCorrectness/AnswerSimilarity wait for R3,
    # which wires the embeddings client into the worker).
    # R3: embedding-based metrics appended to the reference route when
    # EVAL_EMBEDDINGS_METRICS_ENABLED=true (see _build_ragas_embeddings).
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerSimilarity,
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
except ImportError:  # pragma: no cover
    SingleTurnSample = None
    Faithfulness = None
    LLMContextPrecisionWithoutReference = None
    FactualCorrectness = None
    LLMContextPrecisionWithReference = None
    LLMContextRecall = None
    AnswerCorrectness = None
    AnswerSimilarity = None
    ResponseRelevancy = None


def _retrieved_context_texts(raw: Any) -> list[str]:
    """Normalize DB/enqueue shapes: JSONB may come back as str; items may be dict or plain str."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(str(item.get("text", "") or ""))
        elif isinstance(item, str):
            out.append(item)
        else:
            out.append(str(item))
    return [t for t in out if t.strip()]


def _build_ragas_embeddings() -> Any | None:
    """Build a RAGAS embeddings wrapper from the active text-embedding config.

    Reuses tools.vectorizer's provider selection (EMBEDDING_PROVIDER auto/qwen/
    openai/zhipu/openrouter and the per-provider model/key/base_url), wrapped in
    ragas' LangchainEmbeddingsWrapper. Returns None (with a warning log) when no
    provider is usable or construction fails — the caller degrades to the
    LLM-only metric panel instead of failing the evaluation job.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        from . import vectorizer
    except Exception as exc:
        logger.warning("[RAGAS] embeddings metrics disabled — client import failed: {}", exc)
        return None
    api = vectorizer.get_available_api()
    try:
        if api == "qwen":
            inner = OpenAIEmbeddings(
                api_key=config.qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model=config.embedding_model,
            )
        elif api == "openai":
            inner = OpenAIEmbeddings(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                model=(config.openai_embedding_model or "text-embedding-3-small").strip(),
            )
        elif api == "zhipu":
            inner = OpenAIEmbeddings(
                api_key=config.zhipu_api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4",
                model=(config.zhipu_embedding_model or "embedding-3").strip(),
            )
        elif api == "openrouter":
            inner = OpenAIEmbeddings(
                api_key=config.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model=(config.openrouter_embedding_model or "nvidia/nemotron-3-embed-1b:free").strip(),
            )
        else:
            logger.warning(
                "[RAGAS] embeddings metrics disabled — no usable text-embedding provider "
                "(set EMBEDDING_PROVIDER plus its API key)"
            )
            return None
        return LangchainEmbeddingsWrapper(inner)
    except Exception as exc:
        logger.warning("[RAGAS] embeddings metrics disabled — client construction failed: {}", exc)
        return None


async def _score_job(job: dict[str, Any]) -> dict[str, float]:
    """Run RAGAS metrics (caller must ensure ragas is enabled and imports succeeded).

    R2 metric routing (doc §6.2): jobs carrying a ``reference`` get the
    reference-based panel (context_recall / context_precision_with_reference /
    factual_correctness); reference-free jobs keep the legacy panel
    (context_precision_without_reference). Faithfulness runs on both — it is
    the constant cross-route baseline.

    R3: when the job carries a reference AND config
    ``eval_embeddings_metrics_enabled`` (EVAL_EMBEDDINGS_METRICS_ENABLED) is
    true, three embedding-based metrics are appended (answer_correctness /
    answer_similarity / answer_relevancy). If the embeddings client cannot be
    built (no key / unsupported provider / ragas too old), log a warning and
    degrade to the 4-metric LLM panel — never fail the job for this.
    """
    if SingleTurnSample is None or Faithfulness is None or LLMContextPrecisionWithoutReference is None:
        raise RuntimeError("RAGAS metric classes not imported")
    llm = get_llm(
        model_name=config.ragas_llm_model,
        temperature=0.0,
        ragas_strip_json_fence=True,
    )
    reference = str(job.get("reference") or "").strip()
    sample = SingleTurnSample(
        user_input=job["query"],
        response=job["answer"],
        retrieved_contexts=_retrieved_context_texts(job.get("context_json")),
        reference=reference or None,
    )
    scores: dict[str, float] = {
        "faithfulness": float(await Faithfulness(llm=llm).single_turn_ascore(sample)),
    }
    if reference:
        scores["context_precision_with_reference"] = float(
            await LLMContextPrecisionWithReference(llm=llm).single_turn_ascore(sample)
        )
        scores["context_recall"] = float(await LLMContextRecall(llm=llm).single_turn_ascore(sample))
        scores["factual_correctness"] = float(
            await FactualCorrectness(llm=llm).single_turn_ascore(sample)
        )
        if config.eval_embeddings_metrics_enabled:
            if AnswerCorrectness is None or AnswerSimilarity is None or ResponseRelevancy is None:
                logger.warning(
                    "[RAGAS] EVAL_EMBEDDINGS_METRICS_ENABLED=true but embedding metric classes "
                    "are unavailable in the installed ragas; degrading to the 4 LLM metrics"
                )
            else:
                embeddings = _build_ragas_embeddings()
                if embeddings is None:
                    logger.warning(
                        "[RAGAS] EVAL_EMBEDDINGS_METRICS_ENABLED=true but no embeddings client "
                        "could be built; degrading to the 4 LLM metrics"
                    )
                else:
                    scores["answer_correctness"] = float(
                        await AnswerCorrectness(llm=llm, embeddings=embeddings).single_turn_ascore(sample)
                    )
                    scores["answer_similarity"] = float(
                        await AnswerSimilarity(embeddings=embeddings).single_turn_ascore(sample)
                    )
                    scores["answer_relevancy"] = float(
                        await ResponseRelevancy(llm=llm, embeddings=embeddings).single_turn_ascore(sample)
                    )
    else:
        scores["context_precision"] = float(
            await LLMContextPrecisionWithoutReference(llm=llm).single_turn_ascore(sample)
        )
    return scores


async def run_pending_evaluations(limit: int | None = None) -> dict[str, Any]:
    jobs = await list_pending_evaluation_jobs(limit or config.ragas_batch_size)
    processed = 0
    failed = 0
    skipped = 0
    for job in jobs:
        try:
            if not config.ragas_enabled:
                await complete_evaluation_job(
                    job["id"],
                    skipped_reason="RAGAS_ENABLED=false; set RAGAS_ENABLED=true in .env, restart API, then enqueue new jobs",
                )
                skipped += 1
                continue
            if (
                SingleTurnSample is None
                or Faithfulness is None
                or LLMContextPrecisionWithoutReference is None
            ):
                await complete_evaluation_job(
                    job["id"],
                    skipped_reason="ragas import failed; install ragas (see pyproject.toml)",
                )
                skipped += 1
                continue
            scores = await _score_job(job)
            for name, value in scores.items():
                tracer.score_trace(job["trace_id"], name=name, value=value)
            await complete_evaluation_job(job["id"], scores=scores)
            processed += 1
        except Exception as exc:
            logger.exception("[RAGAS] Failed to evaluate job %s", job["id"])
            await complete_evaluation_job(job["id"], error=str(exc))
            failed += 1
    if skipped and not processed and not failed:
        logger.warning(
            "[RAGAS] {} job(s) marked skipped (no LLM/RAGAS run). {}",
            skipped,
            "Enable RAGAS_ENABLED or fix ragas install.",
        )
    return {"processed": processed, "failed": failed, "skipped": skipped}
