"""Asset store abstraction for multimodal page/figure images (Part 2, T1.1).

Asset keys are backend-independent and are the ONLY image reference persisted in
Milvus / API contracts: ``{document_id}/{name}`` (e.g. ``9901/page_0.jpg``).
Phase 1 persists to local disk under ``MULTIMODAL_PAGES_DIR``; phase 2 (MinIO)
implements the same protocol with identical keys so migration touches neither
Milvus points nor API consumers (docs/MULTIMODAL_RAG_PART2_DESIGN.md §6).

Path-traversal defence is intentionally cohesive here: callers only ever pass
untrusted ``(document_id, name)`` pairs (HTTP query params), so validation lives
next to the filesystem access instead of at every call site.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from core.config import config
from loguru import logger

# name: basename with extension only — no separators, no "..", no leading dot.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class AssetKeyError(ValueError):
    """Raised when a document_id/name pair fails validation (HTTP 400 upstream)."""


def validate_asset_key(document_id: int, name: str) -> tuple[int, str]:
    """Validate an asset key pair; returns the normalized (document_id, name)."""
    if isinstance(document_id, bool):  # bool is an int subclass — reject explicitly
        raise AssetKeyError(f"invalid document_id: {document_id!r}")
    try:
        doc_id = int(document_id)
    except (TypeError, ValueError) as exc:
        raise AssetKeyError(f"invalid document_id: {document_id!r}") from exc
    if doc_id <= 0:
        raise AssetKeyError(f"invalid document_id: {document_id!r}")
    clean = str(name or "").strip()
    if not clean or len(clean) > 200 or not _NAME_PATTERN.match(clean):
        raise AssetKeyError(f"invalid asset name: {name!r}")
    return doc_id, clean


class MultimodalAssetStore(Protocol):
    def save(self, document_id: int, name: str, data: bytes) -> str:
        """Persist bytes; returns the backend-independent asset key."""
        ...

    def open(self, document_id: int, name: str) -> bytes:
        """Return stored bytes; KeyError when the asset is missing."""
        ...

    def delete_document(self, document_id: int) -> None:
        """Remove every asset of one document (idempotent)."""
        ...


class LocalFileAssetStore:
    """Phase-1 store: ``MULTIMODAL_PAGES_DIR/{document_id}/{name}`` on local disk."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or config.multimodal_pages_dir)

    def _resolve(self, document_id: int, name: str) -> Path:
        doc_id, clean = validate_asset_key(document_id, name)
        doc_dir = (self.root / str(doc_id)).resolve()
        target = (doc_dir / clean).resolve()
        # Defence in depth: the regex already forbids separators, but resolve()
        # + prefix check also neutralizes symlink tricks and exotic platforms.
        if not str(target).startswith(str(doc_dir) + os.sep) and target != doc_dir:
            raise AssetKeyError(f"asset path escapes document dir: {name!r}")
        return target

    def save(self, document_id: int, name: str, data: bytes) -> str:
        doc_id, clean = validate_asset_key(document_id, name)
        doc_dir = self.root / str(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        target = self._resolve(doc_id, clean)
        target.write_bytes(data)
        return f"{doc_id}/{clean}"

    def open(self, document_id: int, name: str) -> bytes:
        target = self._resolve(document_id, name)
        if not target.is_file():
            raise KeyError(f"{document_id}/{name}")
        return target.read_bytes()

    def delete_document(self, document_id: int) -> None:
        doc_dir = self.root / str(int(document_id))
        if doc_dir.is_dir():
            for child in doc_dir.iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
            doc_dir.rmdir()
            logger.info("[AssetStore] deleted asset dir {}", doc_dir)

    def document_bytes(self, document_id: int) -> int:
        """Total stored bytes for one document (CLI summary/water-level reporting)."""
        doc_dir = self.root / str(int(document_id))
        if not doc_dir.is_dir():
            return 0
        return sum(f.stat().st_size for f in doc_dir.iterdir() if f.is_file())


class MinioAssetStore:
    """Phase-2 store: MinIO bucket with IDENTICAL object keys (``{document_id}/{name}``).

    Key isomorphism is the phase boundary contract: migration moves bytes only,
    Milvus points / API contracts / frontend stay untouched (§6). Failures on
    the MinIO path raise raw SDK exceptions — the API layer maps connection
    errors to 502 and missing objects to 404.
    """

    def __init__(self) -> None:
        from minio import Minio

        self._client = Minio(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_secure,
        )
        self.bucket = config.minio_bucket
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)
            logger.info("[AssetStore] created bucket {}", self.bucket)

    def save(self, document_id: int, name: str, data: bytes) -> str:
        import io

        doc_id, clean = validate_asset_key(document_id, name)
        key = f"{doc_id}/{clean}"
        self._client.put_object(
            self.bucket, key, io.BytesIO(data), len(data), content_type="image/jpeg"
        )
        return key

    def open(self, document_id: int, name: str) -> bytes:
        doc_id, clean = validate_asset_key(document_id, name)
        try:
            response = self._client.get_object(self.bucket, f"{doc_id}/{clean}")
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as exc:
            code = getattr(exc, "code", "")
            if code == "NoSuchKey":
                raise KeyError(f"{document_id}/{name}") from exc
            raise

    def delete_document(self, document_id: int) -> None:
        from minio.deleteobjects import DeleteObject

        doc_id = int(document_id)
        objects = list(
            self._client.list_objects(self.bucket, prefix=f"{doc_id}/", recursive=True)
        )
        if not objects:
            return
        errors = self._client.remove_objects(
            self.bucket, [DeleteObject(obj.object_name) for obj in objects]
        )
        for err in errors:
            logger.warning("[AssetStore] minio delete error: {}", err)

    def document_bytes(self, document_id: int) -> int:
        doc_id = int(document_id)
        return sum(
            obj.size or 0
            for obj in self._client.list_objects(self.bucket, prefix=f"{doc_id}/", recursive=True)
        )


@lru_cache(maxsize=1)
def get_asset_store() -> MultimodalAssetStore:
    backend = (config.multimodal_asset_store or "local").strip().lower()
    if backend == "local":
        return LocalFileAssetStore()
    if backend == "minio":
        return MinioAssetStore()
    raise ValueError(
        f"Unsupported MULTIMODAL_ASSET_STORE: {config.multimodal_asset_store!r} (local | minio)"
    )
