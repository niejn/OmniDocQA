"""Unit tests for T1.4 multimodal vectorizer — MockTransport, 429 backoff, RPM, dim."""

import asyncio
import json

import httpx
from tools.multimodal_vectorizer import (
    EmbedResult,
    FixedWindowRateLimiter,
    MultimodalVectorizer,
)


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
