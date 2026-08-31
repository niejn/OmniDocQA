"""Embedding helpers used by the node-centric RAG pipeline."""

from __future__ import annotations

import asyncio
import random
import time
from typing import List, Optional

from loguru import logger
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from core.config import config

from .rag_stage_log import current_request_id, log_rag

# DashScope OpenAI-compatible embeddings enforce a small per-request input count for
# text-embedding-v3 (see provider docs; LangChain uses 6 for v3).
_QWEN_EMBEDDING_BATCH_CAP = 6
# Single-input length must be in [1, 8192] (chars/tokens per API); larger payloads 400.
_QWEN_MAX_INPUT_CHARS = 8192

# Zhipu (BigModel) Embedding-3 via OpenAI-compatible endpoint: a single request caps at
# 64 texts and ~3072 tokens; 8000 chars is a conservative per-input safety margin.
_ZHIPU_EMBEDDING_BATCH_CAP = 64
_ZHIPU_MAX_INPUT_CHARS = 8000
# OpenRouter embeddings: OpenAI-compatible (https://openrouter.ai/api/v1). The free
# nvidia/nemotron-3-embed-1b model does not accept a `dimensions` param; output dim is
# fixed at 2048 (per model card; 32K token context). Set EMBEDDING_DIMENSION=2048 to match.
_OPENROUTER_EMBEDDING_BATCH_CAP = 64


def get_available_api() -> Optional[str]:
    mode = (config.embedding_provider or "auto").strip().lower()
    if mode == "openai":
        if config.openai_api_key:
            return "openai"
        logger.warning("[Vectorizer] EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is missing")
        return None
    if mode == "qwen":
        if config.qwen_api_key:
            return "qwen"
        logger.warning("[Vectorizer] EMBEDDING_PROVIDER=qwen but QWEN_API_KEY is missing")
        return None
    if mode == "zhipu":
        if config.zhipu_api_key:
            return "zhipu"
        logger.warning("[Vectorizer] EMBEDDING_PROVIDER=zhipu but ZHIPU_API_KEY is missing")
        return None
    if mode == "openrouter":
        if config.openrouter_api_key:
            return "openrouter"
        logger.warning("[Vectorizer] EMBEDDING_PROVIDER=openrouter but OPENROUTER_API_KEY is missing")
        return None
    if mode not in ("auto", ""):
        logger.warning("[Vectorizer] Unknown EMBEDDING_PROVIDER={!r}, using auto", mode)
    if config.qwen_api_key:
        return "qwen"
    if config.openai_api_key:
        return "openai"
    if config.zhipu_api_key:
        return "zhipu"
    if config.openrouter_api_key:
        return "openrouter"
    if config.deepseek_api_key:
        logger.warning("[Vectorizer] DeepSeek does not provide embeddings")
        return None
    return None


def _max_embedding_input_chars(api_type: Optional[str]) -> int:
    if api_type == "qwen":
        return _QWEN_MAX_INPUT_CHARS
    if api_type == "zhipu":
        return _ZHIPU_MAX_INPUT_CHARS
    if api_type == "openrouter":
        # Convert the model's token budget into a conservative character budget
        # when no model-specific tokenizer is available.
        token_budget_chars = int(
            config.openrouter_embedding_max_input_tokens
            * config.openrouter_embedding_chars_per_token
        )
        return max(
            1,
            min(int(config.openrouter_embedding_safe_chars), token_budget_chars),
        )
    return max(1, int(config.embedding_safe_chars))


def get_embedding_safe_chars(api_type: Optional[str] = None) -> int:
    """Return the active provider's conservative pre-embedding character budget."""
    return _max_embedding_input_chars(api_type or get_available_api())


def _normalize_embedding_text(text: str, api_type: Optional[str]) -> str:
    """Truncate to provider limits; empty text is invalid for Qwen/OpenAI embeddings."""
    cap = _max_embedding_input_chars(api_type)
    t = (text or "")[:cap]
    return t if t.strip() else " "


def _embedding_chunk_size(api_type: Optional[str]) -> int:
    if api_type == "qwen":
        return min(_QWEN_EMBEDDING_BATCH_CAP, max(1, config.embedding_batch_size))
    if api_type == "zhipu":
        return max(1, min(config.embedding_batch_size, _ZHIPU_EMBEDDING_BATCH_CAP))
    if api_type == "openrouter":
        return max(1, min(config.embedding_batch_size, _OPENROUTER_EMBEDDING_BATCH_CAP))
    if api_type == "openai":
        return max(1, min(config.embedding_batch_size, 200))
    return 8


def _embed_request_kwargs(api_type: Optional[str]) -> dict:
    """Extra kwargs for the OpenAI-compatible embeddings call, per provider."""
    if api_type == "zhipu":
        dim = config.embedding_dimension
        if not (256 <= dim <= 2048):
            logger.warning(
                "[Vectorizer] Zhipu Embedding-3 dimension must be in [256, 2048]; got {}",
                dim,
            )
        return {"dimensions": dim}
    if api_type == "openrouter":
        return {"encoding_format": "float"}
    return {}


def _build_client_and_model() -> tuple[AsyncOpenAI, str] | tuple[None, None]:
    api_type = get_available_api()
    if api_type == "qwen":
        return (
            AsyncOpenAI(
                api_key=config.qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=120.0,
            ),
            config.embedding_model,
        )
    if api_type == "openai":
        return (
            AsyncOpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                timeout=120.0,
            ),
            (config.openai_embedding_model or "text-embedding-3-small").strip(),
        )
    if api_type == "zhipu":
        return (
            AsyncOpenAI(
                api_key=config.zhipu_api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4",
                timeout=120.0,
            ),
            (config.zhipu_embedding_model or "embedding-3").strip(),
        )
    if api_type == "openrouter":
        return (
            AsyncOpenAI(
                api_key=config.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=120.0,
            ),
            (config.openrouter_embedding_model or "nvidia/nemotron-3-embed-1b:free").strip(),
        )
    return None, None


async def generate_embedding(text: str) -> Optional[List[float]]:
    api = get_available_api()
    _, model = _build_client_and_model()
    t0 = time.perf_counter()
    results = await generate_embeddings_batch([text])
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    emb = results[0] if results else None
    if current_request_id():
        log_rag(
            "embedding",
            provider=api,
            model=model,
            text_len=len(text or ""),
            dim=len(emb) if emb else 0,
            ok=bool(emb),
            latency_ms=elapsed_ms,
        )
    return emb


async def generate_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    client, model = _build_client_and_model()
    if client is None or model is None:
        return [None] * len(texts)

    api_type = get_available_api()
    chunk_size = _embedding_chunk_size(api_type)
    inputs = [_normalize_embedding_text(text, api_type) for text in texts]
    if not inputs:
        return []

    async def _call_batch(batch: list[str]) -> List[Optional[List[float]]]:
        response = await client.embeddings.create(
            model=model, input=batch, **_embed_request_kwargs(api_type)
        )
        embeddings = [item.embedding for item in response.data]
        if len(embeddings) != len(batch):
            return [None] * len(batch)
        return embeddings

    async def _embed_slice(batch: list[str]) -> List[Optional[List[float]]]:
        retries = 3
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                output = await _call_batch(batch)
                if output and any(item is not None for item in output):
                    return output
            except Exception as exc:
                last_err = exc
                backoff = min(10.0, 2.0**attempt) + random.random()
                logger.warning(
                    "[Vectorizer] Batch embedding attempt {}/{} failed: {}. Backoff {:.2f}s",
                    attempt + 1,
                    retries,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

        if last_err is not None:
            logger.error("[Vectorizer] Batch embedding failed after retries: {}", last_err)
            if isinstance(last_err, (APIConnectionError, APITimeoutError)) or (
                "connection" in str(last_err).lower()
            ):
                api = get_available_api()
                if api == "qwen" and config.openai_api_key:
                    logger.error(
                        "[Vectorizer] 无法连接 DashScope（通义嵌入）。若网络受限，请在 .env 设置 "
                        "EMBEDDING_PROVIDER=openai，并保留 OPENAI_API_KEY。"
                    )
                elif api == "qwen":
                    logger.error(
                        "[Vectorizer] 无法连接 DashScope。请检查网络/代理，或配置可用的 OPENAI_API_KEY 并设置 "
                        "EMBEDDING_PROVIDER=openai。"
                    )
                elif api == "zhipu":
                    logger.error(
                        "[Vectorizer] 无法连接智谱 BigModel。请检查 ZHIPU_API_KEY 或网络/代理。"
                    )
                elif api == "openrouter":
                    logger.error(
                        "[Vectorizer] 无法连接 OpenRouter。请检查 OPENROUTER_API_KEY 或网络/代理。"
                    )

        results: List[Optional[List[float]]] = []
        for text in batch:
            safe = _normalize_embedding_text(text, api_type)
            try:
                response = await client.embeddings.create(
                    model=model, input=safe, **_embed_request_kwargs(api_type)
                )
                results.append(response.data[0].embedding)
            except Exception as exc:
                logger.warning("[Vectorizer] Per-item embedding failed: {}", exc)
                results.append(None)
                await asyncio.sleep(0.2 + random.random() * 0.3)
        return results

    out: List[Optional[List[float]]] = []
    for start in range(0, len(inputs), chunk_size):
        slice_ = inputs[start : start + chunk_size]
        out.extend(await _embed_slice(slice_))
        if start + chunk_size < len(inputs):
            await asyncio.sleep(0.05 + random.random() * 0.05)

    return out
