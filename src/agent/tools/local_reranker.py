"""Local sentence-transformers CrossEncoder reranker (Qwen3-Reranker), Bocha-parity interface.

Borrows the loading/scoring recipe from ``tools/qwen_reranker.py``:
- optional BitsAndBytes 4bit/8bit quantization on CUDA (default ``none``, CPU-safe)
- sigmoid activation over the model's yes/no logits as the relevance score

Pairs are ``(query, text|text_preview)`` — same text extraction as ``BochaReranker``.
Model load is lazy and cached; a failed load is remembered so unavailable
requests escalate to the fallback reranker instead of retrying the load.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from core.config import config
from loguru import logger

from .rag_stage_log import log_rag


class LocalReranker:
    """sentence-transformers CrossEncoder reranker with lazy model load."""

    def __init__(self, model: Any | None = None) -> None:
        # An injected model (tests / preloaded instance) skips the heavy load path.
        self._model: Any | None = model
        self._load_error: str | None = None
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True when a model instance is ready (injected or already loaded)."""

    def describe_config(self) -> dict[str, Any]:
        """Runtime diagnostics for health/spec endpoints (mirrors BochaReranker)."""
        return {
            "backend": "local",
            "configured": True,
            "model": config.local_reranker_model,
            "max_length": config.local_reranker_max_length,
            "batch_size": config.local_reranker_batch_size,
            "quantization": config.local_reranker_quantization,
            "model_loaded": self._model is not None,
            "load_error": self._load_error,
        }

    def ensure_loaded(self) -> bool:
        """Public warm-up hook (thread-safe); True when the model is ready."""
        return self._ensure_model() is not None

    def _ensure_model(self) -> Any | None:
        """Lazily load the CrossEncoder; exactly once even under concurrency."""
        with self._load_lock:
            if self._model is not None or self._load_error is not None:
                return self._model
            return self._load_qwen_model()

    def _load_qwen_model(self) -> Any | None:
        """Caller holds _load_lock; loads the Qwen reranker exactly once (failures cached)."""
        try:
            # Heavy imports stay lazy so importing this module remains cheap.
            import torch
            from sentence_transformers import CrossEncoder

            from .qwen_reranker import load_cross_encoder

            model = load_cross_encoder(
                config.local_reranker_model,
                max_length=config.local_reranker_max_length,
                quantization=config.local_reranker_quantization,
                torch_module=torch,
                cross_encoder_class=CrossEncoder,
            )
            self._model = model
            logger.info(
                "[LocalReranker] loaded {} (max_length={}, quantization={})",
                config.local_reranker_model,
                config.local_reranker_max_length,
                config.local_reranker_quantization,
            )
        except Exception as exc:  # pragma: no cover - depends on optional deps/network
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("[LocalReranker] 模型加载失败: {}", self._load_error)
        return self._model

    @staticmethod
    def _activation_fn() -> Any:
        """Sigmoid over yes/no logits; None when torch is unavailable (fake models in tests)."""
        try:
            import torch

            return torch.nn.Sigmoid()
        except Exception:  # pragma: no cover - torch is a project dependency
            return None


    def _predict_scores(self, model: Any, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, doc) pairs on a worker thread; sigmoid probability per pair."""
        kwargs: dict[str, Any] = {
            "batch_size": config.local_reranker_batch_size,
            "show_progress_bar": False,
        }
        activation = self._activation_fn()
        if activation is not None:
            kwargs["activation_fn"] = activation
        scores = model.predict(pairs, **kwargs)
        return [float(score) for score in scores]

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
        out_stats: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Score and re-order candidates (Bocha contract: rerank_score + stats mode)."""
        stats = out_stats if out_stats is not None else {}
        stats.clear()
        stats.update(
            {
                "step": "local_rerank",
                "candidates_in": len(candidates),
                "candidates_out": 0,
                "mode": "pending",
            }
        )
        t0 = time.perf_counter()
        try:
            if not candidates:
                stats.update({"mode": "skipped_empty_candidates", "candidates_out": 0})
                return []

            model = self._model if self._model is not None else self._ensure_model()
            if model is None:
                # Truncate-result + unavailable mode: the composite escalates to the
                # fallback reranker when one exists, otherwise this order stands.
                out = candidates[:top_n]
                stats.update(
                    {
                        "mode": "local_model_unavailable",
                        "error": self._load_error,
                        "candidates_out": len(out),
                        "fallback": "truncate_fusion_order",
                    }
                )
                return out

            doc_texts = [
                str(item.get("text") or item.get("text_preview") or "") for item in candidates
            ]
            pairs = [(query, text) for text in doc_texts]
            scores = await asyncio.to_thread(self._predict_scores, model, pairs)

            reranked: list[dict[str, Any]] = []
            for item, score in zip(candidates, scores, strict=True):
                row = dict(item)
                row["rerank_score"] = round(score, 6)
                reranked.append(row)
            reranked.sort(key=lambda row: row["rerank_score"], reverse=True)
            out = reranked[:top_n]
            stats.update(
                {
                    "mode": "local_success",
                    "candidates_out": len(out),
                    "model": config.local_reranker_model,
                }
            )
            return out
        except Exception as exc:
            out = candidates[:top_n]
            stats.update(
                {
                    "mode": "local_error_fallback",
                    "error": str(exc),
                    "candidates_out": len(out),
                    "fallback": "truncate_fusion_order",
                }
            )
            logger.warning("[LocalReranker] 预测失败，使用融合序截断: {}", exc)
            return out
        finally:
            log_rag(
                "rerank_local",
                mode=stats.get("mode"),
                candidates_in=stats.get("candidates_in"),
                candidates_out=stats.get("candidates_out"),
                model=config.local_reranker_model,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )


local_reranker = LocalReranker()
