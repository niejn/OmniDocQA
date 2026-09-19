#!/usr/bin/env python3
"""Multimodal asset migration — local disk ↔ MinIO, byte-verified (phase 2, §11).

The store boundary contract: asset KEYS are identical on both sides
(``{document_id}/{name}``), so migration is a pure byte move — Milvus points,
API contracts and the frontend are untouched. Default direction local→minio;
``--reverse`` pulls minio→local (rollback).

Verification: every file's size is compared; ``--sample N`` files (default 5)
are additionally compared byte-for-byte. Non-zero exit on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from core.config import config  # noqa: E402
from tools.multimodal_asset_store import (  # noqa: E402
    LocalFileAssetStore,
    MinioAssetStore,
)


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _iter_local_files() -> list[tuple[int, str, Path]]:
    root = Path(config.multimodal_pages_dir)
    if not root.is_dir():
        return []
    out: list[tuple[int, str, Path]] = []
    for doc_dir in sorted(root.iterdir()):
        if doc_dir.is_dir() and doc_dir.name.isdigit():
            for f in sorted(doc_dir.iterdir()):
                if f.is_file() and f.name != "ingest_report.json":
                    out.append((int(doc_dir.name), f.name, f))
    return out


def _iter_minio_files() -> list[tuple[int, str]]:
    store = MinioAssetStore()
    out: list[tuple[int, str]] = []
    for obj in store._client.list_objects(store.bucket, recursive=True):
        doc_part, _, name = obj.object_name.partition("/")
        if doc_part.isdigit() and name:
            out.append((int(doc_part), name))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate multimodal assets local↔minio (byte-verified)")
    parser.add_argument("--reverse", action="store_true", help="minio → local (rollback)")
    parser.add_argument("--sample", type=int, default=5, help="Byte-compare N random files (default 5)")
    parser.add_argument("--delete-source", action="store_true", help="Delete source files after verification")
    args = parser.parse_args()

    local = LocalFileAssetStore()
    minio = MinioAssetStore()
    rng = random.Random(20260919)

    if not args.reverse:
        files = _iter_local_files()
        print(json.dumps({"direction": "local->minio", "files": len(files)}))
        migrated = 0
        for doc_id, name, path in files:
            data = path.read_bytes()
            minio.save(doc_id, name, data)
            migrated += 1
        # size check on everything, byte check on a sample
        size_mismatch: list[str] = []
        sample = rng.sample(files, min(args.sample, len(files)))
        byte_mismatch: list[str] = []
        for doc_id, name, path in files:
            remote = minio.open(doc_id, name)
            if len(remote) != path.stat().st_size:
                size_mismatch.append(f"{doc_id}/{name}")
        for doc_id, name, path in sample:
            if _md5(minio.open(doc_id, name)) != _md5(path.read_bytes()):
                byte_mismatch.append(f"{doc_id}/{name}")
        result = {
            "migrated": migrated,
            "size_mismatch": size_mismatch,
            "byte_sample": [f"{d}/{n}" for d, n, _ in sample],
            "byte_mismatch": byte_mismatch,
            "verified": not size_mismatch and not byte_mismatch,
        }
        if args.delete_source and result["verified"]:
            for _d, _n, path in files:
                path.unlink()
            print(json.dumps({"deleted_source_files": len(files)}))
    else:
        files = _iter_minio_files()
        print(json.dumps({"direction": "minio->local", "files": len(files)}))
        migrated = 0
        for doc_id, name in files:
            data = minio.open(doc_id, name)
            local.save(doc_id, name, data)
            migrated += 1
        size_mismatch: list[str] = []
        sample = rng.sample(files, min(args.sample, len(files)))
        byte_mismatch: list[str] = []
        for doc_id, name in files:
            lp = Path(config.multimodal_pages_dir) / str(doc_id) / name
            if not lp.is_file() or lp.stat().st_size != len(minio.open(doc_id, name)):
                size_mismatch.append(f"{doc_id}/{name}")
        for doc_id, name in sample:
            lp = Path(config.multimodal_pages_dir) / str(doc_id) / name
            if _md5(lp.read_bytes()) != _md5(minio.open(doc_id, name)):
                byte_mismatch.append(f"{doc_id}/{name}")
        result = {
            "migrated": migrated,
            "size_mismatch": size_mismatch,
            "byte_sample": [f"{d}/{n}" for d, n in sample],
            "byte_mismatch": byte_mismatch,
            "verified": not size_mismatch and not byte_mismatch,
        }
        if args.delete_source and result["verified"]:
            # Rollback keeps the bucket contents (cheap storage); clean the
            # bucket explicitly with scripts/multimodal_assets_gc.py semantics
            # or drop the bucket in the MinIO console when confident.
            print(json.dumps({"note": "reverse mode keeps minio objects; clean the bucket manually if desired"}))

    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["verified"] else 1)


if __name__ == "__main__":
    main()
