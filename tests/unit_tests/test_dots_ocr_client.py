"""Unit tests for T1.2 dots.ocr client — layout JSON parsing, md conversion, fallback."""

import pytest
from tools.dots_ocr_client import (
    ParsedPage,
    layout_to_md,
    parse_layout_json,
)


def _cell(category: str, text: str = "", bbox=None) -> dict:
    return {"bbox": bbox or [10, 20, 110, 40], "category": category, "text": text}


def test_parse_layout_json_plain_array():
    cells = parse_layout_json('[{"bbox":[1,2,3,4],"category":"Text","text":"你好"}]')
    assert cells == [{"bbox": [1.0, 2.0, 3.0, 4.0], "category": "Text", "text": "你好"}]


def test_parse_layout_json_code_fence_and_prose():
    raw = '```json\n[{"bbox":[0,0,5,5],"category":"Title","text":"第一章"}]\n```'
    assert parse_layout_json(raw)[0]["category"] == "Title"
    prose = 'The layout is: [{"bbox":[0,0,5,5],"category":"Text","text":"x"}] end.'
    assert parse_layout_json(prose)[0]["text"] == "x"


def test_parse_layout_json_invalid_returns_none():
    assert parse_layout_json("not json at all") is None
    assert parse_layout_json("") is None
    assert parse_layout_json("[{bad}]") is None


def test_parse_layout_json_non_numeric_bbox_returns_none():
    # Non-numeric bbox = unparseable candidate (contract: None, no exception,
    # no half-parsed layout with garbage coordinates).
    raw = '[{"bbox":["a","b","c","d"],"category":"Text","text":"x"}]'
    assert parse_layout_json(raw) is None


def test_layout_to_md_all_categories():
    cells = [
        _cell("Page-header", "Header Noise"),
        _cell("Title", "第一章 Apache Flink 概述"),
        _cell("Text", "大数据处理引擎。", bbox=[0, 0, 100, 20]),
        _cell("Formula", "E = mc^2"),
        _cell("Table", "<table><tr><td>1</td></tr></table>"),
        _cell("Picture"),
        _cell("Page-footer", "3"),
    ]
    md = layout_to_md(cells, page_no=2)
    assert "Header Noise" not in md and "3" != md.strip()  # header/footer skipped
    assert "# 第一章 Apache Flink 概述" in md  # Title carries a real H1 prefix
    assert "大数据处理引擎。" in md
    assert "E = mc^2" in md  # formula latex kept
    assert "<table>" in md  # table HTML inlined
    assert "![image](image_p2_0.jpg)" in md  # picture placeholder with unique name


def test_layout_to_md_section_header_prefix():
    md = layout_to_md([_cell("Section-header", "1.1 什么是 Flink")])
    assert md.startswith("## 1.1 什么是 Flink")


def test_layout_to_md_text_and_list_item_have_no_prefix():
    md = layout_to_md([_cell("Text", "正文。"), _cell("List-item", "要点")])
    assert md == "正文。\n\n要点"


def test_layout_to_md_headers_chunker_compatible():
    """Title/Section-header prefixes must satisfy the chunker's ^#{1,3}\\s rule."""
    from tools.multimodal_chunker import _HEADER_RE

    assert _HEADER_RE.match(layout_to_md([_cell("Title", "章")]))
    assert _HEADER_RE.match(layout_to_md([_cell("Section-header", "节")]))


def test_layout_to_md_multiple_pictures_unique_names():
    cells = [_cell("Picture"), _cell("Picture")]
    md = layout_to_md(cells, page_no=5)
    assert "image_p5_0.jpg" in md and "image_p5_1.jpg" in md


def test_layout_to_md_empty_and_none():
    assert layout_to_md(None) == ""
    assert layout_to_md([]) == ""


def test_fitz_fallback_page_prefix():
    page = ParsedPage(page_no=0, md_content="## 第 1 页\n\n正文", category="fitz")
    assert page.md_content.startswith("## 第 1 页")
    assert page.category == "fitz"


def test_crop_pictures_maps_coordinates():
    from tools.dots_ocr_client import crop_pictures

    pytest.importorskip("PIL")
    from PIL import Image

    page = ParsedPage(
        page_no=0,
        md_content="",
        input_height=100,
        input_width=100,
        layout=[_cell("Picture", bbox=[10, 10, 50, 50]), _cell("Text", "t")],
    )
    img = Image.new("RGB", (200, 400), color=(255, 0, 0))  # scale_x=2, scale_y=4
    crops = crop_pictures(page, img)
    assert set(crops) == {"image_p0_0.jpg"}
    crop = crops["image_p0_0.jpg"]
    assert crop[:2] == b"\xff\xd8"  # JPEG magic


def test_smart_resize_identity_when_already_factor_multiple():
    from tools.dots_ocr_client import smart_resize

    assert smart_resize(196, 196) == (196, 196)
    assert smart_resize(56, 112) == (56, 112)


def test_smart_resize_rounds_to_factor():
    from tools.dots_ocr_client import smart_resize

    # Nearest 28-multiple per axis (A4 @ dpi=200 ≈ 1654x2339).
    assert smart_resize(1654, 2339) == (1652, 2352)
    assert smart_resize(200, 200) == (196, 196)


def test_smart_resize_enlarges_to_min_pixels():
    from tools.dots_ocr_client import _MIN_PIXELS, smart_resize

    w, h = smart_resize(10, 10)
    assert w % 28 == 0 and h % 28 == 0
    assert w * h >= _MIN_PIXELS


def test_smart_resize_shrinks_to_max_pixels():
    from tools.dots_ocr_client import _MAX_PIXELS, smart_resize

    w, h = smart_resize(100000, 100000)
    assert w % 28 == 0 and h % 28 == 0
    assert w * h <= _MAX_PIXELS


def test_smart_resize_rejects_non_positive():
    from tools.dots_ocr_client import smart_resize

    with pytest.raises(ValueError):
        smart_resize(0, 100)


def test_crop_pictures_rescales_from_smart_resize_input_space():
    """End-to-end bbox remap: input space = smart_resize output, page = original render."""
    import io

    from tools.dots_ocr_client import crop_pictures, smart_resize

    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (200, 200), color=(0, 255, 0))
    in_w, in_h = smart_resize(img.width, img.height)  # (196, 196) != image size
    assert (in_w, in_h) != img.size
    page = ParsedPage(
        page_no=1,
        md_content="",
        input_width=in_w,
        input_height=in_h,
        layout=[_cell("Picture", bbox=[49, 49, 98, 98])],
    )
    crops = crop_pictures(page, img)
    # scale = 200/196: bbox 49..98 maps exactly to 50..100 → 50px crop.
    crop = Image.open(io.BytesIO(crops["image_p1_0.jpg"]))
    assert crop.size == (50, 50)


def test_layout_to_md_heading_newline_collapsed_to_one_heading():
    # A newline inside Title/Section-header text would split the cell into a
    # fake heading boundary for the chunker's ^#{1,3}\s scanner (review P2-9).
    md = layout_to_md([_cell("Title", "第一章\nApache Flink 概述")])
    assert md == "# 第一章 Apache Flink 概述"

    md2 = layout_to_md([_cell("Section-header", "1.1\n什么是 Flink")])
    assert md2 == "## 1.1 什么是 Flink"


def test_layout_to_md_non_heading_keeps_newlines():
    # Only heading-prefixed categories are collapsed; body text keeps its lines.
    md = layout_to_md([_cell("Text", "第一行\n第二行")])
    assert md == "第一行\n第二行"
