"""Helpers for RAGAS + LangChain: models often wrap JSON in ```json fences, which breaks Pydantic JSON parsing."""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI


def strip_markdown_json_fence(text: str) -> str:
    """Remove common ``` / ```json wrappers so ``model_validate_json`` succeeds."""
    if not text:
        return text
    s = text.strip()
    if not s.startswith("```"):
        return text
    lines = s.split("\n")
    if not lines:
        return text
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _sanitize_llm_result(result: LLMResult) -> None:
    """Strip fences from every generation text (covers stream + non-stream paths)."""
    import os

    for gen_list in result.generations:
        for g in gen_list:
            raw = getattr(g, "text", None) or ""
            if not raw:
                continue
            if os.environ.get("RAGAS_DEBUG_RAW"):
                print(f"[RAGAS-DEBUG] generation text: {repr(raw)[:200]}", file=sys.stderr, flush=True)
            cleaned = strip_markdown_json_fence(raw)
            if cleaned != raw:
                g.text = cleaned


class RagasSanitizingChatOpenAI(ChatOpenAI):
    """ChatOpenAI that strips markdown code fences so RAGAS ``model_validate_json`` works.
    LangChain may satisfy requests via ``_agenerate`` or ``_astream`` + merge; overriding
    only ``_agenerate`` misses the streaming path, so we sanitize at ``generate`` /
    ``agenerate`` boundaries.

    Also disables reasoning ("thinking") for the judge LLM: ark plan-endpoint models
    are reasoning models whose chain-of-thought can exhaust max_tokens, returning an
    empty ``content`` (finish_reason=length) which breaks every RAGAS pydantic parse.
    Judge prompts only need direct JSON output.
    """

    def __init__(self, **kwargs: Any) -> None:
        extra = {"thinking": {"type": "disabled"}}
        existing = dict(kwargs.pop("extra_body", None) or {})
        kwargs["extra_body"] = {**existing, **extra}
        super().__init__(**kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> LLMResult:
        result = super().generate(*args, **kwargs)
        _sanitize_llm_result(result)
        return result

    async def agenerate(self, *args: Any, **kwargs: Any) -> LLMResult:
        result = await super().agenerate(*args, **kwargs)
        _sanitize_llm_result(result)
        return result
