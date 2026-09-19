"""Image description for multimodal ingestion (T1.3) — two-layer strategy.

Layer 1 (base, always present, zero cost): caption + adjacent context text
stitched from the section body — guaranteed non-placeholder content even when
the VLM is down.

Layer 2 (enhancement, ``MULTIMODAL_DESCRIBE_MODE=vlm`` default): the picture's
jpg is sent to the description VLM with surrounding context to read the
in-image content (numbers/trends/extremes — dots.ocr omits Picture text, so
in-image information only exists in pixels). VLM failure degrades to the base
layer, never blocks vectorization.

Model notes (design §2 measured records): doubao-seed-*-lite ships with
thinking enabled — disabled via ``extra_body`` for cheap batch describing;
GLM-family models reject that parameter, so it is only sent to doubao models.
"""

from __future__ import annotations

from core.config import config
from loguru import logger

_CONTEXT_CHARS = 200  # tail/head chars of prev/next context per side
_MAX_DESC_CHARS = 300

_PROMPT_TEMPLATE = """前文内容: {prev}
后文内容: {next}

请根据以上上下文和图片内容，生成对该图片的简洁描述，描述内容长度最好不超过300个汉字。
要求：读出图内的关键信息（数字、趋势、极值、结构），不要翻译图片原文，直接输出描述正文。"""


def base_description(prev_text: str, next_text: str) -> str:
    """Layer-1 description from adjacent context only (free, always available)."""
    prev = (prev_text or "").strip()[-_CONTEXT_CHARS:]
    next_ = (next_text or "").strip()[:_CONTEXT_CHARS]
    parts: list[str] = []
    if prev:
        parts.append(f"前文：{prev}")
    if next_:
        parts.append(f"后文：{next_}")
    return " ".join(parts).strip()


def _build_client():
    from openai import OpenAI

    base_url = (config.multimodal_vlm_base_url or config.openai_base_url or "").strip() or None
    api_key = config.multimodal_vlm_api_key or config.openai_api_key or "0"
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)


def vlm_description(
    image_data_uri: str,
    prev_text: str,
    next_text: str,
    *,
    client=None,
) -> str | None:
    """Layer-2 in-image reading; None on failure (caller falls back to base)."""
    if not image_data_uri:
        return None
    prompt = _PROMPT_TEMPLATE.format(
        prev=(prev_text or "").strip()[-_CONTEXT_CHARS:] or "（无）",
        next=(next_text or "").strip()[:_CONTEXT_CHARS] or "（无）",
    )
    extra_body: dict = {}
    model = (config.multimodal_vlm_model or "").strip()
    # doubao-seed* defaults to thinking on (measured: non-empty reasoning_content);
    # GLM models 400 on the disable parameter — send it to doubao only.
    if model.lower().startswith("doubao"):
        extra_body = {"thinking": {"type": "disabled"}}
    try:
        client = client or _build_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=1024,
            extra_body=extra_body or None,
        )
        content = (response.choices[0].message.content or "").strip()
        return content[:_MAX_DESC_CHARS * 2] or None
    except Exception as exc:
        logger.warning("[MmVlm] describe failed ({}): {}", model, exc)
        return None


def describe_image(
    image_data_uri: str,
    prev_text: str,
    next_text: str,
    *,
    client=None,
) -> tuple[str, str]:
    """Two-layer describe. Returns (description, mode_used): mode vlm|context|fallback."""
    mode = (config.multimodal_describe_mode or "vlm").strip().lower()
    base = base_description(prev_text, next_text)
    if mode != "vlm":
        return base, "context"
    enhanced = vlm_description(image_data_uri, prev_text, next_text, client=client)
    if enhanced:
        return enhanced, "vlm"
    return base, "fallback"
