"""Document library API — ``/agent/api/documents`` router (T2.3 + MM-4 + §8.5).

Endpoints (§8): ask / collections / page-image / filters / sets CRUD, the
MM-4 chapter-browsing endpoint, and the §8.5.2/§8.5.3 upload API:
upload / documents inventory / document delete / dynamic collections /
testset generate + evaluate jobs. ask_api is untouched (zero-change
boundary); the router is registered in api/server.py.

Error contract (§15): 400 bad asset name · 404 unknown set/image/document ·
422 contract violations & filter typo guard (missing lists in detail) ·
502 upstream model/store failures.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from core.config import config
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field

from ..multimodal_cleanup import DocumentCascadeError
from .document_service import (
    DocumentAskError,
    ask_documents,
    classify_upload_failure,
    create_dynamic_collection,
    create_set,
    delete_document,
    delete_set,
    get_chapter_chunks,
    get_collections,
    get_filters,
    get_job,
    get_page_image,
    list_documents,
    list_sets_with_staleness,
    resolve_multimodal_collection,
    resolve_testset_path,
    start_evaluation_job,
    start_testset_job,
    upload_multimodal_pdf,
)

router = APIRouter(prefix="/api/documents", tags=["Document Library"])


def _http_error(exc: DocumentAskError) -> HTTPException:
    detail: Any = {"message": str(exc)}
    if getattr(exc, "missing", None):
        detail["missing"] = exc.missing
    return HTTPException(status_code=exc.status_code, detail=detail)


class DocumentFilters(BaseModel):
    books: list[str] | None = None
    chapters: list[int] | None = None
    kinds: list[Literal["text", "image"]] | None = None


class DocumentAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    collection: str = Field(default="multimodal")  # text | multimodal | registered dynamic (§8.5.3)
    top_k: int | None = Field(default=None, ge=1, le=50)
    generate_answer: bool = False
    filters: DocumentFilters | None = None
    set_id: str | None = None


class CreateSetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    filter: DocumentFilters | None = None
    chunk_ids: list[str] | None = None


@router.post("/ask")
async def ask_documents_endpoint(request: DocumentAskRequest) -> dict:
    try:
        return await ask_documents(
            question=request.question,
            collection=request.collection,
            top_k=request.top_k,
            generate_answer=request.generate_answer,
            filters=request.filters.model_dump(exclude_none=True) if request.filters else None,
            set_id=request.set_id,
        )
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        logger.exception("[Documents] ask failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc


@router.get("/collections")
async def collections_endpoint() -> dict:
    return await get_collections()


@router.get("/page-image")
def page_image_endpoint(
    document_id: int = Query(...),
    name: str = Query(..., description="asset basename, e.g. page_0.jpg"),
):
    # Sync endpoint: the asset store does file/MinIO IO — FastAPI runs `def`
    # handlers on the threadpool so the event loop is never blocked (review A1).
    try:
        data = get_page_image(document_id, name)
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:  # network/store failures upstream → 502 (review A2)
        logger.exception("[Documents] page image failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc
    return Response(content=data, media_type="image/jpeg")


@router.get("/filters")
async def filters_endpoint() -> dict:
    try:
        return await get_filters()
    except Exception as exc:
        logger.exception("[Documents] filters failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc


@router.get("/sets")
async def list_sets_endpoint() -> dict:
    try:
        return {"sets": await list_sets_with_staleness()}
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        logger.exception("[Documents] list sets failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc


@router.post("/sets")
async def create_set_endpoint(request: CreateSetRequest) -> dict:
    filter_json = request.filter.model_dump(exclude_none=True) if request.filter else None
    has_filter = bool(filter_json and any(filter_json.values()))
    has_chunks = request.chunk_ids is not None
    if has_filter == has_chunks:  # both or neither
        raise HTTPException(
            status_code=422,
            detail={"message": "pass exactly one of filter or chunk_ids"},
        )
    try:
        created = await create_set(
            name=request.name,
            filter_json=filter_json if has_filter else None,
            chunk_ids=request.chunk_ids if has_chunks else None,
        )
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    return {"set": created}


@router.delete("/sets/{set_id}")
async def delete_set_endpoint(set_id: str) -> dict:
    try:
        await delete_set(set_id)
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:  # review A2: upstream store failure → 502
        logger.exception("[Documents] delete set failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc
    return {"deleted": set_id}


@router.get("/chapters/{document_id}/chunks")
async def chapter_chunks_endpoint(
    document_id: int,
    kind: Literal["text", "image"] | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict:
    try:
        return await get_chapter_chunks(document_id, kind=kind, page=page, page_size=page_size)
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        logger.exception("[Documents] chapter chunks failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc


# ── §8.5.2 upload / inventory / delete + §8.5.3 dynamic collections ─────


def _upload_reject(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail={"message": detail})


@router.post("/upload")
async def upload_document_endpoint(
    file: UploadFile = File(...),
    collection: str | None = Form(default=None),
) -> dict:
    """Multipart upload (field ``file``) → shared six-step ingest → UploadResult.

    422 matrix: non-PDF · > MULTIMODAL_UPLOAD_MAX_MB · > MAX_PAGES (from the
    ingest core's ValueError) · unknown collection. Store failures → 502.
    """
    filename = file.filename or ""
    try:
        if not filename.lower().endswith(".pdf"):
            raise _upload_reject(f"only .pdf uploads are supported, got {filename!r}")
        max_bytes = max(1, int(config.multimodal_upload_max_mb)) * 1024 * 1024
        declared = getattr(file, "size", None)
        if declared is not None and int(declared) > max_bytes:
            raise _upload_reject(
                f"upload of {int(declared)} bytes exceeds the {max_bytes}-byte limit "
                f"(MULTIMODAL_UPLOAD_MAX_MB={config.multimodal_upload_max_mb})"
            )
        payload = await file.read()
        if len(payload) > max_bytes:
            raise _upload_reject(
                f"upload of {len(payload)} bytes exceeds the {max_bytes}-byte limit "
                f"(MULTIMODAL_UPLOAD_MAX_MB={config.multimodal_upload_max_mb})"
            )
        tmp_dir = tempfile.mkdtemp(prefix="mm_upload_")
        pdf_path = Path(tmp_dir) / (Path(filename).name or "upload.pdf")
        await asyncio.to_thread(pdf_path.write_bytes, payload)
        try:
            return await upload_multimodal_pdf(pdf_path=pdf_path, collection=collection)
        finally:
            # Remove the whole mkdtemp dir, not just the pdf (review P2-3: the
            # otherwise-empty directory used to leak per upload).
            await asyncio.to_thread(shutil.rmtree, tmp_dir, True)
    except HTTPException:
        raise
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        status = classify_upload_failure(exc)
        logger.exception("[Documents] upload failed ({})", status)
        raise HTTPException(status_code=status, detail={"message": str(exc)[:300]}) from exc


@router.get("/documents")
async def list_documents_endpoint() -> list[dict]:
    """Ingested multimodal documents: BARE array, document_id DESC (frontend contract)."""
    try:
        return await list_documents()
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        logger.exception("[Documents] list documents failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc


@router.delete("/documents/{document_id}")
async def delete_document_endpoint(document_id: int) -> dict:
    try:
        await delete_document(document_id)
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    except DocumentCascadeError as exc:
        logger.error("[Documents] cascade failed at {}: {}", exc.step, exc)
        # Same {"message": ...} shape as the other endpoints (review P2-4).
        raise HTTPException(status_code=502, detail={"message": f"cascade failed at {exc.step}"}) from exc
    except Exception as exc:
        logger.exception("[Documents] delete document failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc
    return {"deleted": True}


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    embedding_provider: str | None = None
    description: str | None = None


@router.post("/collections", status_code=201)
async def create_collection_endpoint(request: CreateCollectionRequest) -> dict:
    """§8.5.3: create a dynamic Milvus collection and register it (201 on success)."""
    try:
        return await create_dynamic_collection(
            name=request.name,
            embedding_provider=request.embedding_provider,
            description=request.description,
        )
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        logger.exception("[Documents] create collection failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc


# ── §8.5.4 testset generation / evaluation jobs ──────────────────────────


class GenerateTestsetRequest(BaseModel):
    collection: str | None = None
    testset_size: int = Field(..., ge=1, le=100)


class EvaluateRequest(BaseModel):
    collection: str | None = None
    testset_path: str = Field(..., min_length=1)


@router.post("/generate-testset", status_code=202)
async def generate_testset_endpoint(request: GenerateTestsetRequest) -> dict:
    """Validate → background question drafting (T2.5 core) → 202 {job_id}."""
    try:
        target = await resolve_multimodal_collection(request.collection)
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    try:
        job_id = await start_testset_job(collection=target, testset_size=request.testset_size)
    except Exception as exc:
        logger.exception("[Documents] start testset job failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc
    return {"job_id": job_id}


@router.get("/testset/{job_id}")
async def testset_job_endpoint(job_id: str) -> dict:
    """Job status: running | completed (questions list) | failed (error)."""
    try:
        return get_job(job_id)
    except DocumentAskError as exc:
        raise _http_error(exc) from exc


@router.post("/evaluate", status_code=202)
async def evaluate_endpoint(request: EvaluateRequest) -> dict:
    """Whitelist-validated testset → background hard-assertion eval → 202 {job_id}."""
    try:
        resolved = resolve_testset_path(request.testset_path)
        target = await resolve_multimodal_collection(request.collection)
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
    try:
        job_id = await start_evaluation_job(collection=target, testset_path=resolved)
    except Exception as exc:
        logger.exception("[Documents] start evaluate job failed")
        raise HTTPException(status_code=502, detail={"message": str(exc)[:300]}) from exc
    return {"job_id": job_id}
