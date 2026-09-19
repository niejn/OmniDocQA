"""Multimodal PDF chunker (T1.3) — LangChain-free self-implementation.

Effect-replicates the reference ``MarkdownDirSplitter`` pipeline per page of
dots.ocr Markdown (docs/MULTIMODAL_RAG_PART2_DESIGN.md §7-2):

1. Header-boundary split on H1-H3 (stateful across pages: a page that starts
   without a heading inherits the previous page's closing hierarchy — v1
   decision §14.1-3, courseware sections routinely span pages).
2. Picture placeholders (``![image](image_pN_i.jpg)`` from ``layout_to_md``)
   are extracted into standalone ``image`` chunks; body text continues
   picture-free.
3. Text chunks over ``text_chunk_size`` chars get semantic splitting: split
   into sentences, embed, cut where adjacent-sentence cosine distance exceeds
   the percentile breakpoint (LangChain SemanticChunker "percentile" parity).
   ``embed_fn=None`` (dry-run/tests) falls back to deterministic fixed-size
   splitting.
4. Every chunk carries the completed title hierarchy ``H1 --> H2 --> H3``.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from .dots_ocr_client import ParsedPage

# ![alt](image_p3_1.jpg) — only our own layout_to_md placeholders are extracted.
_PICTURE_RE = re.compile(r"!\[[^\]]*\]\((image_p\d+_\d+\.jpg)\)")
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_RE = re.compile(r"[^。！？!?.;；\n]+[。！？!?.;；\n]?")

EmbedBatchFn = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass
class MmChunk:
    """One multimodal chunk (text or standalone image)."""

    kind: str  # "text" | "image"
    page_no: int
    title: str  # completed hierarchy "H1 --> H2 --> H3"
    text: str  # body text; for image chunks the description (filled by the VLM step)
    category: str = "Text"  # layout category / "fitz" on the fallback path
    image_name: str | None = None  # asset basename for image chunks
    image_ref: str | None = None  # "{document_id}/{name}" asset key (set by the pipeline)
    truncated: bool = False  # embedding input hit the safe-char cap
    metadata: dict = field(default_factory=dict)


def _split_by_headers(md: str) -> list[tuple[int, str, str]]:
    """Split one page's md on H1-H3 boundaries; returns [(level, header, body)]."""
    sections: list[tuple[int, str, str]] = []
    current_level = 0
    current_header = ""
    body_lines: list[str] = []
    for line in md.splitlines():
        match = _HEADER_RE.match(line)
        if match and len(match.group(1)) <= 3:
            if current_level or any(line.strip() for line in body_lines):
                sections.append((current_level, current_header, "\n".join(body_lines).strip()))
            current_level = len(match.group(1))
            current_header = match.group(2)
            body_lines = []
        else:
            body_lines.append(line)
    if current_level or any(line.strip() for line in body_lines):
        sections.append((current_level, current_header, "\n".join(body_lines).strip()))
    return sections


class _TitleHierarchy:
    """Completed ``H1 --> H2 --> H3`` state carried across pages (§14.1-3)."""

    def __init__(self) -> None:
        self.levels: dict[int, str] = {1: "", 2: "", 3: ""}

    def update(self, level: int, header: str) -> None:
        if level == 0:
            return
        self.levels[level] = header.strip()
        for lower in range(level + 1, 4):
            self.levels[lower] = ""

    def title(self) -> str:
        return " --> ".join(v for v in (self.levels[i] for i in (1, 2, 3)) if v)


def _sentences(text: str) -> list[str]:
    return [s for s in (m.group(0) for m in _SENTENCE_RE.finditer(text)) if s.strip()]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if not na or not nb:
        return 1.0
    return 1.0 - dot / (na * nb)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


def _fixed_size_split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


async def _semantic_split(text: str, embed_fn: EmbedBatchFn, size: int, pct: float) -> list[str]:
    """Percentile-breakpoint semantic split (embedding used only as a chunking signal)."""
    sents = _sentences(text)
    if len(sents) <= 1:
        return [text] if text.strip() else []
    vectors = await embed_fn(sents)
    if len(vectors) != len(sents) or any(v is None for v in vectors):
        logger.warning("[Chunker] embed_fn failed; falling back to fixed-size split")
        return _fixed_size_split(text, size)
    distances = [_cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    threshold = _percentile(distances, pct)
    chunks: list[str] = []
    buffer = sents[0]
    for i, dist in enumerate(distances):
        if dist > threshold and len(buffer) >= size // 4:
            chunks.append(buffer)
            buffer = sents[i + 1]
        else:
            buffer += sents[i + 1]
    if buffer.strip():
        chunks.append(buffer)
    return chunks or [text]


def _extract_images(md: str) -> tuple[str, list[str]]:
    """Pull picture placeholders out of a md fragment; returns (clean_text, names)."""
    names = _PICTURE_RE.findall(md)
    clean = _PICTURE_RE.sub("", md)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, names


async def chunk_document(
    pages: list[ParsedPage],
    *,
    embed_fn: EmbedBatchFn | None = None,
    text_chunk_size: int = 1000,
    semantic_percentile: float = 95.0,
) -> list[MmChunk]:
    """Chunk parsed pages into ordered text/image chunks with inherited titles."""
    chunks: list[MmChunk] = []
    hierarchy = _TitleHierarchy()
    for page in pages:
        sections = _split_by_headers(page.md_content)
        # A page with no leading heading inherits the previous page's hierarchy.
        if sections and sections[0][0] == 0:
            _level, header, body = sections.pop(0)
            inherited = hierarchy.title() or header
            await _emit_section(chunks, inherited, body, page, embed_fn, text_chunk_size, semantic_percentile)
        for level, header, body in sections:
            hierarchy.update(level, header)
            await _emit_section(chunks, hierarchy.title(), body, page, embed_fn, text_chunk_size, semantic_percentile)
    return [c for c in chunks if (c.kind == "image" or c.text.strip())]


async def _emit_section(
    chunks: list[MmChunk],
    title: str,
    body: str,
    page: ParsedPage,
    embed_fn: EmbedBatchFn | None,
    size: int,
    pct: float,
) -> None:
    clean_body, image_names = _extract_images(body)
    # Image chunks come first in reading order within the section body.
    for name in image_names:
        chunks.append(
            MmChunk(
                kind="image",
                page_no=page.page_no,
                title=title,
                text="",  # filled by the describe step
                category=page.category,
                image_name=name,
            )
        )
    if not clean_body.strip():
        return
    pieces: list[str]
    if len(clean_body) > size and embed_fn is not None:
        pieces = await _semantic_split(clean_body, embed_fn, size, pct)
    elif len(clean_body) > size:
        pieces = _fixed_size_split(clean_body, size)
    else:
        pieces = [clean_body]
    for piece in pieces:
        chunks.append(
            MmChunk(
                kind="text",
                page_no=page.page_no,
                title=title,
                text=piece.strip(),
                category=page.category,
            )
        )
