"""Reranker selection: local CrossEncoder by default, optional disable.

``RERANKER_BACKEND`` (local | none) picks the reranker. When the local model
cannot even attempt scoring (model load failure), that backend keeps its own
truncate-fusion-order fallback, so candidates are never dropped.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from core.config import config
from loguru import logger

from .local_reranker import LocalReranker


class TruncateReranker:
    """RERANKER_BACKEND=none: keep the fused order, no scoring."""

    def describe_config(self) -> dict[str, Any]:
        return {"backend": "none", "configured": False}

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
        out_stats: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        stats = out_stats if out_stats is not None else {}
        stats.clear()
        out = candidates[:top_n]
        stats.update(
            {
                "step": "truncate_rerank",
                "mode": "skipped_disabled",
                "candidates_in": len(candidates),
                "candidates_out": len(out),
                "fallback": "truncate_fusion_order",
            }
        )
        return out


@lru_cache(maxsize=1)
def get_reranker() -> LocalReranker | TruncateReranker:
    backend = (config.reranker_backend or "local").strip().lower()
    if backend == "none":
        return TruncateReranker()
    if backend != "local":
        # Legacy values (e.g. "bocha", removed with the remote reranker) used to
        # map to local SILENTLY — surface the misconfiguration once.
        logger.warning(
            '[Rerank] unsupported RERANKER_BACKEND={backend!r}; falling back to "local" '
            '(valid values: local | none)',
            backend=backend,
        )
    return LocalReranker()


async def warmup_reranker() -> None:
    """Preload the local reranker model when it is the primary backend.

    Safe to call under any RERANKER_BACKEND; no-op otherwise. Runs the heavy
    load off the event loop so server startup stays responsive.
    """
    import asyncio

    reranker = get_reranker()
    if isinstance(reranker, LocalReranker):
        await asyncio.to_thread(reranker.ensure_loaded)


# Module-level singleton used across the retrieval pipeline.
reranker = get_reranker()
