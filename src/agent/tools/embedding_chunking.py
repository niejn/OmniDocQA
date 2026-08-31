"""Provider-safe text splitting used before embedding and node construction."""

from __future__ import annotations

import re
from dataclasses import replace

from loguru import logger

from .chunk_segmenter import ChunkPayload

DEFAULT_SAFE_CHARS = 3000
DEFAULT_OVERLAP_CHARS = 220
_SENTENCE_BREAK_RE = re.compile(r"(?<=[。！？；：.!?;:])\s*|[\r\n]+")


def _preview(text: str, limit: int = 120) -> str:
    """Return a compact single-line preview for debug logs."""
    return " ".join((text or "").split())[:limit]


def split_text_for_embedding(
    text: str,
    *,
    max_chars: int = DEFAULT_SAFE_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split text at sentence boundaries, with a hard-split fallback."""
    normalized = (text or "").strip()
    if not normalized:
        return []
    limit = max(1, int(max_chars))
    overlap = max(0, min(int(overlap_chars), limit - 1))
    if len(normalized) <= limit:
        return [normalized]

    sentences = [part.strip() for part in _SENTENCE_BREAK_RE.split(normalized) if part.strip()]
    pieces: list[str] = []
    current = ""

    def emit(value: str) -> None:
        if value.strip():
            pieces.append(value.strip())

    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                emit(current)
                current = ""
            start = 0
            while start < len(sentence):
                end = min(start + limit, len(sentence))
                emit(sentence[start:end])
                if end == len(sentence):
                    break
                start = end - overlap
            continue
        next_text = f"{current}\n\n{sentence}" if current else sentence
        if current and len(next_text) > limit:
            emit(current)
            overlap_text = current[-overlap:] if overlap else ""
            current = f"{overlap_text}\n\n{sentence}".strip() if overlap_text else sentence
            if len(current) > limit:
                current = sentence
        else:
            current = next_text
    emit(current)
    return pieces


def split_chunk_payloads(
    chunks: list[ChunkPayload],
    *,
    max_chars: int = DEFAULT_SAFE_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[ChunkPayload]:
    """Apply the same safe split to every pre-built ChunkPayload."""
    out: list[ChunkPayload] = []
    split_count = 0
    added_count = 0
    for chunk in chunks:
        pieces = split_text_for_embedding(
            chunk.text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        if len(pieces) <= 1:
            out.append(chunk)
            continue
        split_count += 1
        added_count += len(pieces) - 1
        logger.info(
            "[ChunkSplit] chunk title={!r} chars={} split into {} chunks",
            chunk.title,
            len(chunk.text),
            len(pieces),
        )
        for part, piece in enumerate(pieces, start=1):
            part_metadata = dict(chunk.metadata)
            out.append(replace(chunk, text=piece, metadata=part_metadata))
            logger.debug(
                "[ChunkSplit] chunk title={!r} part={}/{} chars={} preview={!r}",
                chunk.title,
                part,
                len(pieces),
                len(piece),
                _preview(piece),
            )
    logger.info(
        "[ChunkSplit] processed={} output={} newly_added={} split_inputs={}",
        len(chunks),
        len(out),
        added_count,
        split_count,
    )
    return out
