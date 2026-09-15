"""Reranker selection: local CrossEncoder by default, Bocha remote retained as fallback.

``RERANKER_BACKEND`` (local | bocha | none) picks the primary reranker; the
composite escalates to the next backend only when the primary cannot even
attempt scoring (model load failure / not configured / disabled). Runtime
errors inside a backend keep that backend's own truncate-fusion-order fallback
(same behaviour as the previous Bocha-only pipeline).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from core.config import config

from .bocha_reranker import BochaReranker
from .local_reranker import LocalReranker

# Modes meaning "this backend could not attempt scoring" -> escalate to the next.
_UNAVAILABLE_MODES = {"local_model_unavailable", "skipped_not_configured", "skipped_disabled"}


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


class CompositeReranker:
    """Ordered reranker chain with unavailable-escalation."""

    def __init__(self, chain: list[Any]) -> None:
        self._chain = chain

    def describe_config(self) -> dict[str, Any]:
        return {
            "backend": (config.reranker_backend or "local").strip().lower(),
            "chain": [item.describe_config() for item in self._chain],
        }

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
        out_stats: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        stats = out_stats if out_stats is not None else {}
        for index, item in enumerate(self._chain):
            is_last = index == len(self._chain) - 1
            stage_stats = stats if index == 0 else {}
            out = await item.rerank(
                query=query, candidates=candidates, top_n=top_n, out_stats=stage_stats
            )
            mode = str(stage_stats.get("mode") or "")
            if not is_last and mode in _UNAVAILABLE_MODES:
                stats["escalated_to"] = type(self._chain[index + 1]).__name__
                continue
            if index > 0:
                stats["fallback_stage"] = dict(stage_stats)
            return out
        # Unreachable safeguard: never drop candidates.
        return candidates[:top_n]


@lru_cache(maxsize=1)
def get_reranker() -> CompositeReranker:
    backend = (config.reranker_backend or "local").strip().lower()
    local = LocalReranker()
    bocha = BochaReranker()
    if backend == "bocha":
        chain = [bocha]
    elif backend == "none":
        chain = [TruncateReranker()]
    else:
        chain = [local, bocha] if bocha.enabled else [local]
    return CompositeReranker(chain)


async def warmup_reranker() -> None:
    """Preload the local reranker model when it is the primary backend.

    Safe to call under any RERANKER_BACKEND; no-op otherwise. Runs the heavy
    load off the event loop so server startup stays responsive.
    """
    import asyncio

    composite = get_reranker()
    chain = getattr(composite, "_chain", [])
    first = chain[0] if chain else None
    if isinstance(first, LocalReranker):
        await asyncio.to_thread(first.ensure_loaded)


# Module-level singleton: drop-in replacement for `from .bocha_reranker import reranker`.
reranker = get_reranker()
