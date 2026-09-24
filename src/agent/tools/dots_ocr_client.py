"""dots.ocr / dots.mocr parse client via vLLM, with fitz text fallback (T1.2).

Protocol replicated from the dots.ocr reference implementation (MIT,
docs/MULTIMODAL_RAG_PART2_DESIGN.md §1) without importing the package:

- OpenAI-compatible chat call: one ``image_url`` block (base64 data URI of the
  dpi=200 page render) + a text prompt prefixed with ``<|img|><|imgpad|><|endofimg|>``
  (vLLM v1 adds a stray newline without it).
- The model returns a single JSON array of layout cells
  ``{bbox: [x1,y1,x2,y2], category, text}`` over 11 categories; Picture has no
  text, Formula is LaTeX, Table is HTML, everything else Markdown, in reading
  order.
- ``layout_to_md`` converts cells to per-page Markdown; Pictures become
  ``![image](image_{n}.jpg)`` placeholders that the chunker later extracts.
- Page-level failures degrade per the official "filtered" path: raw response is
  kept as the page md (plain-text semantics), layout is emptied, the batch
  continues.

Insertion crops: dots.ocr only emits bboxes for Picture cells, so the actual
jpg bytes are cropped here from the dpi=200 page render. Before sending, the
page image is pre-scaled to the qwen-vl ``smart_resize`` fit (28-multiple
axes, min/max pixel budget) and that input space is recorded on the page, so
``crop_pictures`` remaps model bboxes back to the rendered pixels with real
(non-1.0) scale factors — the mapping factor is logged on the first page of
the first document (§14.1-1 bbox first-verification).
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from core.config import config
from loguru import logger

from .rag_stage_log import log_rag

# The 11 layout categories of the dots.ocr prompt protocol (§1).
LAYOUT_CATEGORIES = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)

# Qwen-VL vision input fitting (qwen-vl-utils smart_resize parity): images are
# rounded to multiples of 28 under a min/max pixel budget, so vLLM receives a
# pre-fitted image and bbox coordinates live in a known, recorded input space.
_IMAGE_FACTOR = 28
_MIN_PIXELS = 4 * 28 * 28  # 3136 — qwen-vl-utils default
_MAX_PIXELS = 16384 * 28 * 28  # 12,845,056 — qwen-vl-utils default

_PROMPT_LAYOUT_ALL_EN = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

# vLLM v1 newline hack — must prefix the prompt text block (reference inference.py).
_IMG_PAD_PREFIX = "<|img|><|imgpad|><|endofimg|>"

# Chunker-facing picture placeholder inside page md: ![image](image_{n}.jpg)
_PICTURE_PLACEHOLDER = "![image]({name})"


@dataclass
class ParsedPage:
    """One parsed PDF page (dots.ocr path or fitz fallback)."""

    page_no: int
    md_content: str
    page_image_jpg: bytes | None = None
    layout: list[dict] | None = None  # None => fallback/filtered page (no layout info)
    filtered: bool = False
    category: str = "dots_ocr"  # "fitz" on the fallback path
    input_height: int | None = None
    input_width: int | None = None
    error: str | None = None
    picture_crops: dict[str, bytes] = field(default_factory=dict)  # name -> jpg bytes


def _render_page_images(pdf_path: str | Path, dpi: int) -> tuple[list, list[tuple[int, int]]]:
    """Render every page at ``dpi``; returns (PIL images, (w,h) sizes)."""
    import fitz
    from PIL import Image

    images: list = []
    sizes: list[tuple[int, int]] = []
    with fitz.open(str(pdf_path)) as doc:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
            sizes.append((pix.width, pix.height))
    return images, sizes


def _image_to_data_uri(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _jpg_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def smart_resize(
    width: int,
    height: int,
    *,
    factor: int = _IMAGE_FACTOR,
    min_pixels: int = _MIN_PIXELS,
    max_pixels: int = _MAX_PIXELS,
) -> tuple[int, int]:
    """Qwen-VL ``smart_resize`` (qwen-vl-utils parity): round both axes to
    multiples of ``factor`` (28) and clamp the total pixel budget between
    ``min_pixels`` and ``max_pixels``. Returns ``(resized_width, resized_height)``.

    Pure geometry (no image IO) so every branch is unit-testable. Zero/negative
    input and degenerate sub-factor sizes are floored at one factor step.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive image size: {width}x{height}")

    def _round_factor(value: int) -> int:
        return max(factor, round(value / factor) * factor)

    w_bar = _round_factor(width)
    h_bar = _round_factor(height)
    if w_bar * h_bar > max_pixels:
        beta = math.sqrt((width * height) / max_pixels)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
    elif w_bar * h_bar < min_pixels:
        beta = math.sqrt(min_pixels / (width * height))
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
    return w_bar, h_bar


def _resize_image(img, width: int, height: int):
    from PIL import Image

    return img.resize((int(width), int(height)), Image.LANCZOS)


def parse_layout_json(response_text: str) -> list[dict] | None:
    """Extract the layout cell array from a model response; None when unparsable.

    Tolerates code fences and prose around the JSON (official output_cleaner
    behaviour): strips fences first, then falls back to the outermost
    ``[...]`` span.
    """
    if not response_text or not str(response_text).strip():
        return None
    text = str(response_text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    start, end = text.find("["), text.rfind("]")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):  # tolerate {"cells": [...]} wrappers
                for value in data.values():
                    if isinstance(value, list):
                        data = value
                        break
            if not (isinstance(data, list) and all(isinstance(cell, dict) for cell in data)):
                continue
            cells = []
            for cell in data:
                bbox = cell.get("bbox")
                category = str(cell.get("category") or "").strip()
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not category:
                    continue
                cells.append(
                    {
                        # float() inside the try: a non-numeric bbox makes the
                        # whole candidate unparsable (contract: return None),
                        # not a half-parsed layout with garbage coordinates.
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "category": category,
                        "text": str(cell.get("text") or ""),
                    }
                )
            return cells
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def _formula_to_md(text: str) -> str:
    """Wrap a LaTeX formula in a $$ block (reference get_formula_in_markdown, simplified)."""
    text = (text or "").strip()
    if text.startswith("$$") and text.endswith("$$"):
        return text
    if text.startswith("\\[") and text.endswith("\\]"):
        return f"$$\n{text[2:-2].strip()}\n$$"
    if "$" in text or "\\" in text:
        return text if text.startswith("$") else f"$$\n{text}\n$$"
    return text


# Category → markdown heading prefix. The chunker's title hierarchy only
# recognizes ``^#{1,3}\s`` lines, so Title/Section-header cells must carry a
# real heading level (fitz fallback uses ``## 第 N 页``, same style).
_MD_PREFIX_BY_CATEGORY = {
    "Title": "# ",
    "Section-header": "## ",
}


def layout_to_md(cells: list[dict] | None, *, page_no: int = 0, no_page_hf: bool = True) -> str:
    """Convert layout cells to page Markdown (reference layoutjson2md semantics).

    Title/Section-header cells get ``# ``/``## `` markdown prefixes so the
    chunker's header-boundary split and title hierarchy pick them up; Picture
    cells become ``![image](image_{n}.jpg)`` placeholders where ``n`` counts
    pictures across the whole document via ``page_no``-suffixed names
    (``image_p{page_no}_{idx}.jpg``) so names are unique per document without
    cross-page state.
    """
    if cells is None:
        return ""
    items: list[str] = []
    picture_idx = 0
    for cell in cells:
        category = str(cell.get("category") or "")
        if no_page_hf and category in ("Page-header", "Page-footer"):
            continue
        text = str(cell.get("text") or "").strip()
        if category == "Picture":
            name = f"image_p{page_no}_{picture_idx}.jpg"
            picture_idx += 1
            items.append(_PICTURE_PLACEHOLDER.format(name=name))
            continue
        if category == "Formula":
            items.append(_formula_to_md(text))
            continue
        if text:
            prefix = _MD_PREFIX_BY_CATEGORY.get(category)
            if prefix:
                # Heading text must stay ONE line: an embedded newline inside a
                # Title/Section-header cell would split into a fake heading
                # boundary for the chunker's ^#{1,3}\s scanner (review P2-9).
                text = " ".join(text.split())
            items.append(prefix + text if prefix else text)
    return "\n\n".join(items)


def crop_pictures(page: ParsedPage, page_image) -> dict[str, bytes]:
    """Crop Picture cells from the rendered page image into ``page.picture_crops``.

    Coordinates arrive in the model's input space (smart-resized); remap to the
    page-image pixel space with per-axis scale factors. The first page logs the
    mapping factors for the §14.1-1 bbox first-verification.
    """

    if not page.layout:
        return page.picture_crops
    img_w, img_h = page_image.size
    in_w = page.input_width or img_w
    in_h = page.input_height or img_h
    scale_x = img_w / in_w if in_w else 1.0
    scale_y = img_h / in_h if in_h else 1.0
    picture_idx = 0
    for cell in page.layout:
        if str(cell.get("category") or "") != "Picture":
            continue
        name = f"image_p{page.page_no}_{picture_idx}.jpg"
        picture_idx += 1
        x1, y1, x2, y2 = cell["bbox"]
        box = (
            max(0, min(img_w - 1, int(x1 * scale_x))),
            max(0, min(img_h - 1, int(y1 * scale_y))),
            max(1, min(img_w, int(x2 * scale_x))),
            max(1, min(img_h, int(y2 * scale_y))),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        page.picture_crops[name] = _jpg_bytes(page_image.crop(box))
        if page.page_no == 0:
            log_rag(
                "parse_bbox_check",
                page_no=page.page_no,
                bbox_raw=[*cell["bbox"]],
                input_hw=[in_h, in_w],
                page_hw=[img_h, img_w],
                scale=[round(scale_x, 4), round(scale_y, 4)],
                crop_box=list(box),
            )
    return page.picture_crops


class DotsOcrClient:
    """Thin dots.ocr client: health probe, page-parallel parse, fitz fallback."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or config.dot_ocr_base_url or "").strip().rstrip("/")

    def healthy(self, timeout: float = 3.0) -> bool:
        """False when DOT_OCR_BASE_URL is empty or the vLLM /models probe fails."""
        if not self.base_url:
            return False
        try:
            import httpx

            response = httpx.get(f"{self.base_url}/models", timeout=timeout)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("[DotsOcr] health probe failed ({}): {}", self.base_url, exc)
            return False

    def _new_openai_client(self):
        """One OpenAI client per parse batch (httpx pool reused across page threads)."""
        from openai import OpenAI

        return OpenAI(api_key=config.dot_ocr_api_key or "0", base_url=self.base_url, timeout=300.0)

    def _parse_single_page(self, page_image, page_no: int, client) -> ParsedPage:
        page = ParsedPage(page_no=page_no, md_content="", page_image_jpg=_jpg_bytes(page_image))
        try:
            # Qwen-VL smart_resize: send the model a 28-multiple pre-fitted image
            # and record THAT input space — bbox remapping in crop_pictures then
            # has a real (non-1.0) scale back to the rendered page pixels.
            input_w, input_h = smart_resize(page_image.width, page_image.height)
            model_image = (
                page_image
                if (input_w, input_h) == page_image.size
                else _resize_image(page_image, input_w, input_h)
            )
            data_uri = _image_to_data_uri(model_image)
            response = client.chat.completions.create(
                model=config.dot_ocr_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": f"{_IMG_PAD_PREFIX}{_PROMPT_LAYOUT_ALL_EN}"},
                        ],
                    }
                ],
                temperature=config.dot_ocr_temperature,
                max_completion_tokens=config.dot_ocr_max_completion_tokens,
            )
            raw = response.choices[0].message.content or ""
            cells = parse_layout_json(raw)
            if cells is None:
                # Official "filtered" degradation: keep raw response as plain text.
                page.filtered = True
                page.error = "layout JSON parse failed"
                page.md_content = str(raw)[:65000]
                return page
            page.layout = cells
            page.input_width = input_w
            page.input_height = input_h
            page.md_content = layout_to_md(cells, page_no=page_no)
            crop_pictures(page, page_image)
        except Exception as exc:  # single-page failure must not kill the batch
            page.filtered = True
            page.error = str(exc)[:500]
            logger.warning("[DotsOcr] page {} parse failed: {}", page_no, exc)
        return page

    def parse_pdf(self, pdf_path: str | Path, page_images: list | None = None) -> list[ParsedPage]:
        """Parse all pages via vLLM with a thread pool; results sorted by page_no."""
        if page_images is None:
            page_images, _sizes = _render_page_images(pdf_path, config.dot_ocr_dpi)
        total = len(page_images)
        threads = max(1, min(total, int(config.dot_ocr_max_threads)))
        log_rag("parse", backend="dots_ocr", pages=total, threads=threads, dpi=config.dot_ocr_dpi)
        results: dict[int, ParsedPage] = {}
        # One shared OpenAI client for the whole batch: per-page construction
        # used to rebuild an HTTP connection pool for every single page.
        client = self._new_openai_client()
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = {
                pool.submit(self._parse_single_page, img, idx, client): idx
                for idx, img in enumerate(page_images)
            }
            for future in as_completed(futures):
                page = future.result()
                results[page.page_no] = page
        return [results[i] for i in sorted(results)]

    def parse_pdf_fitz_fallback(self, pdf_path: str | Path, page_images: list | None = None) -> list[ParsedPage]:
        """fitz degradation: ``## 第 N 页`` prefixed text + full-page jpg, no layout.

        Without layout information there are no Picture crops (v1 decision,
        v1 §4.2): only text chunks are vectorized; page images are stored for
        display only.
        """
        import fitz

        if page_images is None:
            page_images, _sizes = _render_page_images(pdf_path, config.dot_ocr_dpi)
        pages: list[ParsedPage] = []
        t0 = time.perf_counter()
        with fitz.open(str(pdf_path)) as doc:
            for idx, page in enumerate(doc):
                text = (page.get_text("text") or "").strip()
                md = f"## 第 {idx + 1} 页\n\n{text}" if text else f"## 第 {idx + 1} 页"
                pages.append(
                    ParsedPage(
                        page_no=idx,
                        md_content=md,
                        page_image_jpg=_jpg_bytes(page_images[idx]),
                        layout=None,
                        filtered=False,
                        category="fitz",
                    )
                )
        log_rag(
            "parse",
            backend="fitz_fallback",
            pages=len(pages),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return pages

    def parse(self, pdf_path: str | Path) -> tuple[list[ParsedPage], str]:
        """Parse with automatic degradation. Returns (pages, backend_used).

        Raises ``RuntimeError`` when vLLM is unreachable and fallback is off.
        """
        pdf_path = Path(pdf_path)
        if self.healthy():
            pages = self.parse_pdf(pdf_path)
            return pages, "dots_ocr"
        if config.dot_ocr_fallback_fitz:
            logger.warning(
                "[DotsOcr] vLLM unreachable at {!r}; falling back to fitz text extraction",
                self.base_url or "(unset)",
            )
            return self.parse_pdf_fitz_fallback(pdf_path), "fitz_fallback"
        raise RuntimeError(
            f"dots.ocr vLLM unreachable at {self.base_url!r} and DOT_OCR_FALLBACK_FITZ=false"
        )


def load_page_images(pdf_path: str | Path, dpi: int | None = None) -> tuple[list, list[tuple[int, int]]]:
    """Public helper so the CLI renders page images exactly once for all steps."""
    return _render_page_images(pdf_path, dpi or config.dot_ocr_dpi)
