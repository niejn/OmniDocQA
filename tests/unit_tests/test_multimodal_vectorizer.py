"""Unit tests for T1.4 multimodal vectorizer — MockTransport, 429 backoff, RPM, dim."""

import asyncio
import json

import httpx
import pytest
from core.config import config
from tools.multimodal_vectorizer import (
    EmbedResult,
    FixedWindowRateLimiter,
    MultimodalVectorizer,
)


@pytest.fixture(autouse=True)
def _hermetic_ark_endpoint(monkeypatch: pytest.MonkeyPatch):
    """Pin the ark endpoint/key so the suite passes without a local .env.

    ``_call_ark`` resolves the base URL from config (MULTIMODAL_EMBEDDING_BASE_URL
    or OPENAI_BASE_URL) and raises when both are empty — patch the resolvers so
    no environment file is required.
    """
    monkeypatch.setattr("tools.multimodal_vectorizer._effective_base_url", lambda: "http://ark.test")
    monkeypatch.setattr("tools.multimodal_vectorizer._effective_api_key", lambda: "test-key")


def _ark_response(vector: list[float], single_object: bool = True) -> dict:
    data = {"embedding": vector} if single_object else [({"embedding": vector})]
    return {"data": data, "model": "doubao-embedding-vision"}


def test_text_payload_and_truncation_marker():
    payload = MultimodalVectorizer.text_payload("你好")
    assert payload == [{"type": "text", "text": "你好"}]
    long_text = "字" * 9000
    blocks = MultimodalVectorizer.text_payload(long_text)
    assert len(blocks[0]["text"]) == 8000
    # truncated flag: the embed result carries it (checked in _call via >= cap length)
    assert len(MultimodalVectorizer.text_payload("x" * 7999)[0]["text"]) == 7999


def test_image_payload_shape():
    blocks = MultimodalVectorizer.image_payload("data:image/jpeg;base64,AAA", "描述")
    assert blocks[0]["type"] == "image_url"
    assert blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert blocks[1] == {"type": "text", "text": "描述"}


def test_ark_single_object_and_list_response_shapes():
    async def run(response_body: dict):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json=response_body)

        vectorizer = MultimodalVectorizer(transport=httpx.MockTransport(handler))
        try:
            return await vectorizer.embed_text("q"), captured
        finally:
            await vectorizer.aclose()

    result, captured = asyncio.run(run(_ark_response([0.1, 0.2], single_object=True)))
    assert result.vector == [0.1, 0.2]
    # ark multimodal endpoint path + content-block input shape
    assert captured["body"]["input"] == [{"type": "text", "text": "q"}]
    result2, _ = asyncio.run(run(_ark_response([0.3], single_object=False)))
    assert result2.vector == [0.3]


def test_429_backoff_then_success(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("tools.multimodal_vectorizer.asyncio.sleep", fake_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_ark_response([1.0, 2.0]))

    async def run():
        limiter = FixedWindowRateLimiter(10000, 60)
        vectorizer = MultimodalVectorizer(transport=httpx.MockTransport(handler), limiter=limiter)
        try:
            return await vectorizer.embed_text("hello")
        finally:
            await vectorizer.aclose()

    result = asyncio.run(run())
    assert result.vector == [1.0, 2.0]
    assert len(sleeps) == 2  # exponential backoff happened twice
    assert sleeps[1] > sleeps[0]  # exponential growth


def test_retries_exhausted_returns_error(monkeypatch):
    async def fake_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr("tools.multimodal_vectorizer.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    async def run():
        limiter = FixedWindowRateLimiter(10000, 60)
        vectorizer = MultimodalVectorizer(transport=httpx.MockTransport(handler), limiter=limiter)
        try:
            return await vectorizer.embed_text("hello")
        finally:
            await vectorizer.aclose()

    result = asyncio.run(run())
    assert result.vector is None
    assert "rate limited" in (result.error or "")


def test_429_account_quota_fails_fast_without_backoff(monkeypatch):
    """ark ``AccountQuotaExceeded`` 429 is non-retryable: ONE call, ZERO sleeps,
    and the error carries the provider code + reset time (09-22 finding: six
    retries × ~134s then an opaque 500)."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("tools.multimodal_vectorizer.asyncio.sleep", fake_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": "AccountQuotaExceeded",
                    "message": "You have exceeded the monthly usage quota. "
                    "It will reset at 2026-09-23 23:59:59 +0800 CST.",
                    "type": "TooManyRequests",
                }
            },
        )

    async def run():
        limiter = FixedWindowRateLimiter(10000, 60)
        vectorizer = MultimodalVectorizer(transport=httpx.MockTransport(handler), limiter=limiter)
        try:
            return await vectorizer.embed_text("hello")
        finally:
            await vectorizer.aclose()

    result = asyncio.run(run())
    assert calls["n"] == 1  # failed fast: no retry attempts
    assert sleeps == []  # and no backoff sleeps
    assert result.vector is None
    assert result.quota_exhausted is True
    error = result.error or ""
    assert "AccountQuotaExceeded" in error
    assert "2026-09-23 23:59:59" in error  # ark's reset time reaches the caller


def test_429_transient_exhaustion_skips_terminal_sleep(monkeypatch):
    """Transient 429s retry ``max_retries`` times — the pre-fix bug slept one
    extra backoff AFTER the final attempt before giving up (~69s wasted)."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("tools.multimodal_vectorizer.asyncio.sleep", fake_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": "rate limited"})

    async def run():
        limiter = FixedWindowRateLimiter(10000, 60)
        vectorizer = MultimodalVectorizer(transport=httpx.MockTransport(handler), limiter=limiter)
        try:
            return await vectorizer.embed_text("hello")
        finally:
            await vectorizer.aclose()

    result = asyncio.run(run())
    assert calls["n"] == config.multimodal_embed_max_retries + 1
    assert len(sleeps) == config.multimodal_embed_max_retries  # one sleep per RETRY, none terminal
    assert result.vector is None and result.quota_exhausted is False
    assert "rate limited" in (result.error or "")


def test_http_error_fail_fast():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "image too small"})

    async def run():
        limiter = FixedWindowRateLimiter(10000, 60)
        vectorizer = MultimodalVectorizer(transport=httpx.MockTransport(handler), limiter=limiter)
        try:
            return await vectorizer.embed_text("hello")
        finally:
            await vectorizer.aclose()

    result = asyncio.run(run())
    assert result.vector is None and "400" in (result.error or "")


def test_rpm_window_blocks(monkeypatch):
    clocks = {"now": 0.0}
    sleeps: list[float] = []

    def clock() -> float:
        return clocks["now"]

    async def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        clocks["now"] += seconds

    async def run():
        limiter = FixedWindowRateLimiter(3, 60, clock=clock, sleeper=sleeper)
        for _ in range(5):
            await limiter.acquire()
        return sleeps

    sleeps = asyncio.run(run())
    assert len(sleeps) >= 1  # the 4th+ acquire had to wait for the window
    assert sleeps[0] > 0


def test_rpm_window_concurrent_admits_exactly_limit():
    """10 coroutines racing a fresh 3-RPM window (P1-3 regression).

    Without the internal lock every waiter slept AND reset the window, so a
    window admitted more than ``limit`` requests. With the lock held across
    the whole critical section (wait included), each window admits exactly
    ``limit`` requests and each rollover resets the window exactly once.
    """
    clocks = {"now": 0.0}
    sleeps: list[float] = []

    def clock() -> float:
        return clocks["now"]

    async def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        clocks["now"] += seconds
        await asyncio.sleep(0)  # real suspension point inside the critical section

    async def run():
        limiter = FixedWindowRateLimiter(3, 60, clock=clock, sleeper=sleeper)
        admitted: list[tuple[float, int]] = []

        async def acquire_one() -> None:
            await limiter.acquire()
            admitted.append((limiter.window_start, limiter.count))

        await asyncio.gather(*(acquire_one() for _ in range(10)))
        return limiter, admitted

    limiter, admitted = asyncio.run(run())
    assert len(admitted) == 10  # every coroutine completes (lock wait, no deadlock)
    windows: dict[float, int] = {}
    for window_start, _count in admitted:
        windows[window_start] = windows.get(window_start, 0) + 1
    # Exactly `limit` admissions in every complete window…
    complete = sorted(count for count in windows.values() if count == limiter.limit)
    assert complete == [limiter.limit, limiter.limit, limiter.limit]
    # …and never more than `limit` in any window (the old race over-admitted).
    assert all(count <= limiter.limit for count in windows.values())
    # Each window rollover slept (and thus reset) exactly once.
    assert sleeps == [60.0, 60.0, 60.0]
    assert limiter.count <= limiter.limit


def test_image_embedding_requires_data_uri():
    async def run():
        vectorizer = MultimodalVectorizer(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        try:
            return await vectorizer.embed_image_with_text("", "desc")
        finally:
            await vectorizer.aclose()

    result = asyncio.run(run())
    assert result.vector is None and "empty image" in (result.error or "")


def test_embed_result_defaults():
    result = EmbedResult(vector=None)
    assert result.vector is None and not result.truncated and result.error is None
    assert result.quota_exhausted is False
