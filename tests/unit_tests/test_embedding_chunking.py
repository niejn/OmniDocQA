from tools.chunk_segmenter import ChunkPayload
from tools.embedding_chunking import split_chunk_payloads, split_text_for_embedding


def test_long_text_is_split_without_exceeding_limit():
    text = "第一段内容。\n\n" + ("第二段内容。" * 900)
    pieces = split_text_for_embedding(text, max_chars=1000, overlap_chars=50)

    assert len(pieces) > 1
    assert all(len(piece) <= 1000 for piece in pieces)
    assert pieces[0].startswith("第一段内容")
    assert pieces[-1].endswith("内容。")


def test_chinese_and_english_punctuation_are_sentence_boundaries():
    chinese_pieces = split_text_for_embedding("中文句子。下一句！", max_chars=8, overlap_chars=0)
    english_pieces = split_text_for_embedding("English sentence. Next sentence?", max_chars=20, overlap_chars=0)

    assert chinese_pieces == ["中文句子。", "下一句！"]
    assert english_pieces == ["English sentence.", "Next sentence?"]


def test_chunk_payload_split_preserves_metadata_and_sequence():
    chunk = ChunkPayload(
        text="x" * 2500,
        title="Business",
        metadata={"section_path": ["Business"]},
    )

    result = split_chunk_payloads([chunk], max_chars=1000, overlap_chars=10)

    assert len(result) == 3
    assert all(item.metadata["section_path"] == ["Business"] for item in result)
    assert [item.text for item in result] == ["x" * 1000, "x" * 1000, "x" * 520]
