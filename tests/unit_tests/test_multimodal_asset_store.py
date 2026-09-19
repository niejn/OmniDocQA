"""Unit tests for T1.1 asset store — roundtrip, traversal defence, idempotent delete."""

from pathlib import Path

import pytest
from tools.multimodal_asset_store import (
    AssetKeyError,
    LocalFileAssetStore,
    validate_asset_key,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalFileAssetStore:
    return LocalFileAssetStore(root=tmp_path / "assets")


def test_save_open_roundtrip(store: LocalFileAssetStore):
    key = store.save(9901, "page_0.jpg", b"\xff\xd8jpeg-bytes")
    assert key == "9901/page_0.jpg"
    assert store.open(9901, "page_0.jpg") == b"\xff\xd8jpeg-bytes"
    assert (Path(store.root) / "9901" / "page_0.jpg").is_file()


def test_layout_matches_design_doc(store: LocalFileAssetStore):
    store.save(1, "image_abc123.jpg", b"x")
    assert (Path(store.root) / "1" / "image_abc123.jpg").is_file()


@pytest.mark.parametrize(
    "name",
    ["../escape.jpg", "..", "a/b.jpg", "C:\\evil.jpg", ".hidden", "", " ", "a" * 201],
)
def test_path_traversal_rejected(store: LocalFileAssetStore, name: str):
    with pytest.raises(AssetKeyError):
        store.save(1, name, b"x")
    with pytest.raises(AssetKeyError):
        store.open(1, name)


def test_invalid_document_id_rejected():
    with pytest.raises(AssetKeyError):
        validate_asset_key(-1, "page_0.jpg")
    with pytest.raises(AssetKeyError):
        validate_asset_key("not-an-int", "page_0.jpg")
    with pytest.raises(AssetKeyError):
        validate_asset_key(True, "page_0.jpg")


def test_open_missing_raises_key_error(store: LocalFileAssetStore):
    with pytest.raises(KeyError):
        store.open(42, "missing.jpg")


def test_delete_document_idempotent(store: LocalFileAssetStore):
    store.save(7, "page_0.jpg", b"a")
    store.save(7, "page_1.jpg", b"b")
    store.save(8, "page_0.jpg", b"c")
    store.delete_document(7)
    assert not (Path(store.root) / "7").exists()
    assert store.open(8, "page_0.jpg") == b"c"
    # Idempotent: deleting again is a no-op, not an error.
    store.delete_document(7)
    assert store.document_bytes(8) == 1


def test_document_bytes(store: LocalFileAssetStore):
    store.save(9, "a.jpg", b"1234")
    store.save(9, "b.jpg", b"12")
    assert store.document_bytes(9) == 6
    assert store.document_bytes(404) == 0
