"""Document library API — ``/agent/api/documents`` router (T2.3 + MM-4).

Endpoints (§8): ask / collections / page-image / filters / sets CRUD, plus
the MM-4 chapter-browsing endpoint. ask_api is untouched (zero-change
boundary); the router is registered in api/server.py.

Error contract (§15): 400 bad asset name · 404 unknown set/image ·
422 contract violations & filter typo guard (missing lists in detail) ·
502 upstream model/store failures.
"""

from __future__ import annotations

from typing import Any, Literal

from core.config import config
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field

from .document_service import (
    DocumentAskError,
    ask_documents,
    create_set,
    delete_set,
    get_chapter_chunks,
    get_collections,
    get_filters,
    get_page_image,
    list_sets_with_staleness,
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
    collection: Literal["text", "multimodal"] = "multimodal"
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
    return get_collections()


@router.get("/page-image")
async def page_image_endpoint(
    document_id: int = Query(...),
    name: str = Query(..., description="asset basename, e.g. page_0.jpg"),
):
    try:
        data = get_page_image(document_id, name)
    except DocumentAskError as exc:
        raise _http_error(exc) from exc
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


def default_top_k() -> int:
    return int(config.document_ask_default_top_k)
