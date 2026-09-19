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
    assert "第一章 Apache Flink 概述" in md
    assert "大数据处理引擎。" in md
    assert "E = mc^2" in md  # formula latex kept
    assert "<table>" in md  # table HTML inlined
    assert "![image](image_p2_0.jpg)" in md  # picture placeholder with unique name


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
