"""OpenRouter embedding provider dispatch — no network.

Mirrors test_vectorizer_zhipu.py. Verifies api_type wiring, base_url, model default,
and the OpenAI-compatible request kwargs (encoding_format=float) for
nvidia/nemotron-3-embed-1b:free.
"""

import tools.vectorizer as v
from core.config import config

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _set(monkeypatch, **kw):
    for key, val in kw.items():
        monkeypatch.setattr(config, key, val)


def test_explicit_openrouter_with_key(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="openrouter",
        openrouter_api_key="test-key",
        openrouter_embedding_model="nvidia/nemotron-3-embed-1b:free",
        qwen_api_key=None,
        openai_api_key=None,
        zhipu_api_key=None,
    )
    assert v.get_available_api() == "openrouter"
    client, model = v._build_client_and_model()
    assert client is not None
    assert _OPENROUTER_BASE in str(client.base_url)
    assert model == "nvidia/nemotron-3-embed-1b:free"


def test_openrouter_model_defaults_when_env_empty(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="openrouter",
        openrouter_api_key="test-key",
        openrouter_embedding_model="",
        qwen_api_key=None,
        openai_api_key=None,
        zhipu_api_key=None,
    )
    _, model = v._build_client_and_model()
    assert model == "nvidia/nemotron-3-embed-1b:free"


def test_openrouter_without_key_returns_none(monkeypatch):
    _set(monkeypatch, embedding_provider="openrouter", openrouter_api_key=None)
    assert v.get_available_api() is None


def test_auto_fallback_to_openrouter(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="auto",
        openrouter_api_key="test-key",
        qwen_api_key=None,
        openai_api_key=None,
        zhipu_api_key=None,
    )
    assert v.get_available_api() == "openrouter"


def test_auto_prefers_qwen_over_openrouter(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="auto",
        qwen_api_key="qwen-key",
        openrouter_api_key="test-key",
    )
    assert v.get_available_api() == "qwen"


def test_openrouter_limits_capped(monkeypatch):
    _set(monkeypatch, embedding_batch_size=100, openrouter_embedding_safe_chars=3000)
    assert v._embedding_chunk_size("openrouter") == 64
    assert v._max_embedding_input_chars("openrouter") == 3000


def test_provider_specific_safe_chars_do_not_change_generic_fallback(monkeypatch):
    _set(
        monkeypatch,
        openrouter_embedding_safe_chars=2800,
        embedding_safe_chars=7000,
    )
    assert v.get_embedding_safe_chars("openrouter") == 2800
    assert v.get_embedding_safe_chars("openai") == 7000


def test_openrouter_kwargs_encoding_float(monkeypatch):
    assert v._embed_request_kwargs("openrouter") == {"encoding_format": "float"}


def test_non_openrouter_providers_unaffected(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="openai",
        openai_api_key="openai-key",
        openrouter_api_key="test-key",
    )
    assert v.get_available_api() == "openai"
    assert v._embed_request_kwargs("openai") == {}
