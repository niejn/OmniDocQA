"""Unit tests for T1.3 chunker — headers, inheritance, pictures, semantic split."""

import asyncio

from tools.dots_ocr_client import ParsedPage
from tools.multimodal_chunker import MmChunk, chunk_document


def _page(page_no: int, md: str, category: str = "dots_ocr") -> ParsedPage:
    return ParsedPage(page_no=page_no, md_content=md, category=category)


def _run(coro):
    return asyncio.run(coro)


def test_header_boundary_split():
    md = "# 第一章\n\nFlink 介绍。\n\n## 1.1 什么是 Flink\n\n有界流与无界流。\n\n## 1.2 架构\n\nJobManager。"
    chunks = _run(chunk_document([_page(0, md)]))
    assert [c.title for c in chunks] == [
        "第一章",
        "第一章 --> 1.1 什么是 Flink",
        "第一章 --> 1.2 架构",
    ]
    assert all(c.kind == "text" for c in chunks)


def test_cross_page_title_inheritance():
    pages = [
        _page(0, "# 第一章\n\n开始内容。"),
        _page(1, "无标题续页内容，继承上一页标题。"),  # no heading on this page
        _page(2, "## 1.2 小节\n\n更多内容。"),
    ]
    chunks = _run(chunk_document(pages))
    assert chunks[1].title == "第一章"  # inherited
    assert chunks[2].title == "第一章 --> 1.2 小节"


def test_picture_placeholder_extracted_as_image_chunk():
    md = "# 章\n\n前文。\n\n![image](image_p0_0.jpg)\n\n后文。"
    chunks = _run(chunk_document([_page(0, md)]))
    images = [c for c in chunks if c.kind == "image"]
    texts = [c for c in chunks if c.kind == "text"]
    assert len(images) == 1 and images[0].image_name == "image_p0_0.jpg"
    assert all("image_p0_0.jpg" not in c.text for c in texts)  # removed from body
    assert any("前文。" in c.text for c in texts) and any("后文。" in c.text for c in texts)


def test_semantic_split_called_over_threshold():
    captured: dict = {}

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        captured["texts"] = texts
        return [[float(len(t)), 1.0] for t in texts]

    # Alternating long/short sentences → adjacent-cosine distances spike at
    # boundaries, so the percentile cut reliably produces multiple chunks.
    parts = []
    for i in range(40):
        parts.append(f"这是第{i}个长句子，包含很多字用来撑长度" * 3)
        parts.append("短句。")
    body = "".join(parts)
    assert len(body) > 1000
    chunks = _run(
        chunk_document(
            [_page(0, f"# 章\n\n{body}")],
            embed_fn=fake_embed,
            text_chunk_size=200,
            semantic_percentile=60,
        )
    )
    assert captured["texts"], "embed_fn must be used for oversized chunks"
    text_chunks = [c for c in chunks if c.kind == "text"]
    assert len(text_chunks) > 1  # split happened


def test_embed_failure_falls_back_to_fixed_split():
    async def bad_embed(texts):
        return [None] * len(texts)

    body = "。".join("字" * 20 for _ in range(60)) + "。"  # separable sentences
    chunks = _run(chunk_document([_page(0, f"# 章\n\n{body}")], embed_fn=bad_embed, text_chunk_size=500))
    text_chunks = [c for c in chunks if c.kind == "text"]
    assert len(text_chunks) >= 3
    assert all(len(c.text) <= 500 for c in text_chunks)


def test_no_title_page_single_block():
    chunks = _run(chunk_document([_page(0, "纯文本页面，无标题。")]))
    assert len(chunks) == 1 and chunks[0].kind == "text"


def test_empty_text_only_image_chunks_kept():
    md = "# 章\n\n![image](image_p0_0.jpg)"
    chunks = _run(chunk_document([_page(0, md)]))
    assert [c.kind for c in chunks] == ["image"]


def test_chunk_dataclass_defaults():
    chunk = MmChunk(kind="text", page_no=1, title="t", text="b")
    assert chunk.category == "Text" and chunk.image_name is None and not chunk.truncated
