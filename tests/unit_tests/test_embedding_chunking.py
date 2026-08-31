from tools.chunk_segmenter import ChunkPayload
from tools.embedding_chunking import split_chunk_payloads, split_text_for_embedding


def test_long_text_is_split_without_exceeding_limit():
    text = "第一段内容。\n\n" + ("第二段内容。" * 900)
    pieces = split_text_for_embedding(text, max_chars=1000, overlap_chars=50)

    assert len(pieces) > 1
    assert all(len(piece) <= 1000 for piece in pieces)
    assert pieces[0].startswith("第一段内容")
    assert pieces[-1].endswith("内容。")


def test_chunk_payload_split_preserves_metadata_and_records_parts():
    chunk = ChunkPayload(
        text="x" * 2500,
        title="Business",
        metadata={"section_path": ["Business"]},
    )

    result = split_chunk_payloads([chunk], max_chars=1000, overlap_chars=10)

    assert len(result) == 3
    assert [item.metadata["source_chunk_part"] for item in result] == [1, 2, 3]
    assert all(item.metadata["source_chunk_total"] == 3 for item in result)
    assert all(item.metadata["section_path"] == ["Business"] for item in result)
