"""Provider-safe text splitting used before embedding and node construction."""

from __future__ import annotations

import re
from dataclasses import replace

from .chunk_segmenter import ChunkPayload

DEFAULT_SAFE_CHARS = 3000
DEFAULT_OVERLAP_CHARS = 220
_SENTENCE_BREAK_RE = re.compile(r"(?<=[。！？!?；;：:.])\s+|\n+")


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

    units = [part.strip() for part in _SENTENCE_BREAK_RE.split(normalized) if part.strip()]
    pieces: list[str] = []
    current = ""

    def emit(value: str) -> None:
        if value.strip():
            pieces.append(value.strip())

    for unit in units:
        if len(unit) > limit:
            if current:
                emit(current)
                current = ""
            start = 0
            while start < len(unit):
                end = min(start + limit, len(unit))
                emit(unit[start:end])
                if end == len(unit):
                    break
                start = end - overlap
            continue
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and len(candidate) > limit:
            emit(current)
            carry = current[-overlap:] if overlap else ""
            current = f"{carry}\n\n{unit}".strip() if carry else unit
        else:
            current = candidate
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
    for chunk in chunks:
        pieces = split_text_for_embedding(
            chunk.text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        if len(pieces) <= 1:
            out.append(chunk)
            continue
        total = len(pieces)
        for part, piece in enumerate(pieces, start=1):
            part_metadata = dict(chunk.metadata)
            part_metadata.update({"source_chunk_part": part, "source_chunk_total": total})
            out.append(replace(chunk, text=piece, metadata=part_metadata))
    return out
