"""Multimodal (image+text joint) embedding client (T1.4) — ark-first, httpx direct.

Deliberately NOT reusing ``tools/vectorizer.py``: different model, different
input structure (content-block arrays, not plain strings), different rate-limit
profile (design §4.1 — forced abstraction would couple the two vector spaces).

ark provider (default): POST ``{base}/embeddings/multimodal`` where base
defaults to ``OPENAI_BASE_URL`` (subscription plan endpoint, measured working).
Response ``data`` is a SINGLE object (not the OpenAI array) — both shapes are
handled. Image input is the content-block pair
``[{type: image_url, image_url: {url: data-uri}}, {type: text, text: desc}]``.

dashscope provider (fallback, optional SDK): ``MultiModalEmbedding.call`` with
``{image, text}`` input; only importable when the ``dashscope`` package is
installed.

Rate limiting mirrors the reference ``FixedWindowRateLimiter`` (120 RPM window)
plus exponential 429 backoff (5 tries, base 2.0s). ``MockTransport``-testable:
pass ``transport`` to inject an httpx transport in unit tests.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from functools import lru_cache

import httpx
from core.config import config
from loguru import logger

# Conservative per-input text cap (ark rejects oversized inputs with 400).
_MAX_TEXT_CHARS = 8000
_IMAGE_B64_PLACEHOLDER_MIN_PIXELS = 14  # measured: server rejects images smaller than this


@dataclass
class EmbedResult:
    vector: list[float] | None
    truncated: bool = False
    error: str | None = None


class FixedWindowRateLimiter:
    """Fixed-window RPM limiter (reference parity, async-sleep so the event loop
    is never blocked while waiting for the window to roll over)."""

    def __init__(self, limit: int, window_seconds: int = 60, *, clock=time.monotonic, sleeper=None) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = int(window_seconds)
        self._clock = clock
        self._sleeper = sleeper or asyncio.sleep
        self.window_start = clock()
        self.count = 0

    async def acquire(self) -> None:
        now = self._clock()
        elapsed = now - self.window_start
        if elapsed >= self.window_seconds:
            self.window_start = now
            self.count = 0
        if self.count >= self.limit:
            sleep_sec = self.window_seconds - elapsed
            if sleep_sec > 0:
                logger.debug("[MmVector] rate limit hit, sleeping {:.2f}s", sleep_sec)
                await self._sleeper(sleep_sec)
            self.window_start = self._clock()
            self.count = 0
        self.count += 1


@lru_cache(maxsize=1)
def _sync_limiter() -> FixedWindowRateLimiter:
    return FixedWindowRateLimiter(config.multimodal_embed_rpm)


def _effective_base_url() -> str:
    base = (config.multimodal_embedding_base_url or config.openai_base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("MULTIMODAL_EMBEDDING_BASE_URL unset and OPENAI_BASE_URL empty: no ark endpoint")
    return base


def _effective_api_key() -> str:
    return config.multimodal_embedding_api_key or config.openai_api_key or ""


def _truncate(text: str) -> tuple[str, bool]:
    clean = str(text or "")
    if len(clean) > _MAX_TEXT_CHARS:
        return clean[:_MAX_TEXT_CHARS], True
    return clean, False


class MultimodalVectorizer:
    """Async multimodal embedding client (ark direct / dashscope fallback)."""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None, limiter: FixedWindowRateLimiter | None = None) -> None:
        self._transport = transport
        self._limiter = limiter or _sync_limiter()
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0, transport=self._transport)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── payload builders ──────────────────────────────────────────────
    @staticmethod
    def text_payload(text: str) -> list[dict]:
        clean, _trunc = _truncate(text)
        return [{"type": "text", "text": clean}]

    @staticmethod
    def image_payload(image_data_uri: str, text: str) -> list[dict]:
        clean, _trunc = _truncate(text)
        return [
            {"type": "image_url", "image_url": {"url": image_data_uri}},
            {"type": "text", "text": clean},
        ]

    # ── core call with 429 backoff ────────────────────────────────────
    async def _call_ark(self, input_blocks: list[dict]) -> EmbedResult:
        url = f"{_effective_base_url()}/embeddings/multimodal"
        headers = {"Authorization": f"Bearer {_effective_api_key()}"}
        body = {"model": config.multimodal_embedding_model, "input": input_blocks}
        max_retries = max(0, int(config.multimodal_embed_max_retries))
        last_error: str | None = None
        for attempt in range(max_retries + 1):
            await self._limiter.acquire()
            try:
                response = await self._http().post(url, json=body, headers=headers)
            except Exception as exc:
                last_error = f"request error: {exc}"
                logger.warning("[MmVector] attempt {} network error: {}", attempt + 1, exc)
            else:
                if response.status_code == 429:
                    last_error = "rate limited (429)"
                    backoff = float(config.multimodal_embed_backoff_base) * (2**attempt) * (0.8 + random.random() * 0.4)
                    logger.warning("[MmVector] 429 on attempt {}/{}, backoff {:.2f}s", attempt + 1, max_retries + 1, backoff)
                    await asyncio.sleep(backoff)
                    continue
                if response.status_code != 200:
                    return EmbedResult(vector=None, error=f"HTTP {response.status_code}: {response.text[:300]}")
                payload = response.json()
                data = payload.get("data")
                # ark returns a single object; OpenAI-style returns a list — accept both.
                if isinstance(data, list):
                    data = data[0] if data else None
                vector = (data or {}).get("embedding")
                if not vector:
                    return EmbedResult(vector=None, error=f"no embedding in response: {str(payload)[:300]}")
                truncated = any(len(str(b.get("text") or "")) >= _MAX_TEXT_CHARS for b in input_blocks)
                return EmbedResult(vector=[float(x) for x in vector], truncated=truncated)
        return EmbedResult(vector=None, error=f"retries exhausted: {last_error}")

    async def _call_dashscope(self, input_blocks: list[dict]) -> EmbedResult:
        try:
            import dashscope
        except ImportError:  # optional dependency by design
            return EmbedResult(vector=None, error="provider=dashscope requires the dashscope package")
        text = next((b.get("text") for b in reversed(input_blocks) if b.get("type") == "text"), "")
        image = next((b["image_url"]["url"] for b in input_blocks if b.get("type") == "image_url"), None)
        input_data: list[dict] = [{"text": text}] if not image else [{"image": image, "text": text}]
        await self._limiter.acquire()
        try:
            response = await asyncio.to_thread(
                dashscope.MultiModalEmbedding.call,
                model=config.multimodal_embedding_model,
                input=input_data,
                api_key=config.dashscope_api_key or None,
            )
        except Exception as exc:
            return EmbedResult(vector=None, error=f"dashscope error: {exc}")
        try:
            vector = response.output["embeddings"][0]["embedding"]
            return EmbedResult(vector=[float(x) for x in vector])
        except Exception as exc:
            return EmbedResult(vector=None, error=f"dashscope response parse error: {exc}")

    async def _embed(self, input_blocks: list[dict]) -> EmbedResult:
        provider = (config.multimodal_embedding_provider or "ark").strip().lower()
        if provider == "dashscope":
            return await self._call_dashscope(input_blocks)
        return await self._call_ark(input_blocks)

    # ── public API ────────────────────────────────────────────────────
    async def embed_text(self, text: str) -> EmbedResult:
        return await self._embed(self.text_payload(text))

    async def embed_texts(self, texts: list[str]) -> list[EmbedResult]:
        return [await self.embed_text(t) for t in texts]

    async def embed_image_with_text(self, image_data_uri: str, text: str) -> EmbedResult:
        if not image_data_uri:
            return EmbedResult(vector=None, error="empty image data uri")
        return await self._embed(self.image_payload(image_data_uri, text))

    async def detect_dim(self) -> int:
        """First-call dimension probe (MULTIMODAL_EMBEDDING_DIM=0 path)."""
        result = await self.embed_text("dim probe")
        if result.vector is None:
            raise ValueError(f"multimodal dim probe failed: {result.error}")
        return len(result.vector)


def build_vectorizer(**kwargs) -> MultimodalVectorizer:
    """Constructor hook the CLI/API use (keeps transport injection test-only)."""
    return MultimodalVectorizer(**kwargs)
