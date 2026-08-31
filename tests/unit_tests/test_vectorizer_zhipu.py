"""Zhipu (BigModel) Embedding-3 provider dispatch — no network.

Solidifies the wiring added to tools/vectorizer.py: provider selection,
the OpenAI-compatible client base URL, the dimension kwarg, and the
batch/char caps. The real .env (which may carry a live ZHIPU_API_KEY) is
overridden per test via monkeypatch, so no credentials or network are used.
"""

from core.config import config
import tools.vectorizer as v

_ZHIPU_BASE = "https://open.bigmodel.cn/api/paas/v4"


def _set(monkeypatch, **kw):
    for key, val in kw.items():
        monkeypatch.setattr(config, key, val)


def test_explicit_zhipu_with_key(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="zhipu",
        zhipu_api_key="test-key",
        zhipu_embedding_model="embedding-3",
        embedding_dimension=1024,
        qwen_api_key=None,
        openai_api_key=None,
    )
    assert v.get_available_api() == "zhipu"
    client, model = v._build_client_and_model()
    assert client is not None
    assert _ZHIPU_BASE in str(client.base_url)
    assert model == "embedding-3"


def test_zhipu_model_defaults_when_env_empty(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="zhipu",
        zhipu_api_key="test-key",
        zhipu_embedding_model="",
        embedding_dimension=1024,
    )
    _, model = v._build_client_and_model()
    assert model == "embedding-3"


def test_zhipu_without_key_returns_none(monkeypatch):
    _set(monkeypatch, embedding_provider="zhipu", zhipu_api_key=None)
    assert v.get_available_api() is None


def test_auto_fallback_to_zhipu(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="auto",
        zhipu_api_key="k",
        qwen_api_key=None,
        openai_api_key=None,
    )
    assert v.get_available_api() == "zhipu"


def test_auto_prefers_qwen_over_zhipu(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="auto",
        qwen_api_key="qk",
        zhipu_api_key="zk",
    )
    assert v.get_available_api() == "qwen"


def test_zhipu_limits_capped(monkeypatch):
    _set(monkeypatch, embedding_batch_size=100)
    # API allows at most 64 texts per request
    assert v._embedding_chunk_size("zhipu") == 64
    assert v._max_embedding_input_chars("zhipu") == 8000


def test_zhipu_dimension_passthrough_even_when_out_of_range(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="zhipu",
        zhipu_api_key="k",
        embedding_dimension=9999,
    )
    assert v._embed_request_kwargs("zhipu") == {"dimensions": 9999}


def test_non_zhipu_providers_unaffected(monkeypatch):
    _set(
        monkeypatch,
        embedding_provider="openai",
        openai_api_key="ok",
        zhipu_api_key=None,
    )
    assert v.get_available_api() == "openai"
