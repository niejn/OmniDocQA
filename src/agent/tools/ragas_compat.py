"""ragas<=0.4.3 import compat: stub the removed langchain vertexai module."""

from __future__ import annotations

import sys
import types

_STUB_NAME = "langchain_community.chat_models.vertexai"


def ensure_ragas_importable() -> None:
    """Make ``import ragas`` survive langchain-community>=0.4.

    ragas<=0.4.3 eagerly imports ``langchain_community.chat_models.vertexai``
    (ragas/llms/base.py), which langchain-community>=0.4 removed — a plain
    ``import ragas`` raises ModuleNotFoundError. Upstream PR #2769 makes the
    import lazy but is unmerged (its commit lives on a contributor fork, so
    neither a PyPI release nor a git pin can supply the fix yet). Vertex AI is
    never configured in this stack, so a None ``ChatVertexAI`` stub is inert.

    Call before the first ragas import; delete this module once ragas>=0.4.4
    ships the lazy import.
    """
    if _STUB_NAME in sys.modules:
        return
    try:
        __import__(_STUB_NAME)
    except ImportError:
        stub = types.ModuleType(_STUB_NAME)
        stub.ChatVertexAI = None  # type: ignore[attr-defined]
        sys.modules[_STUB_NAME] = stub
