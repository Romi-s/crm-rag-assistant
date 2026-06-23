import time
from collections import Counter
from typing import Optional

import filetype
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile

from app.agent.graph import qa_graph
from app.config import settings
from app.providers import get_provider
from app.schemas.responses import (
    CitationResponse,
    CollectionStatsResponse,
    IngestResponse,
    QueryResponse,
    RetrievedRef,
)
from app.services.ingest import (
    get_collection,
    ingest_documents,
    ingest_pdf,
    ingest_text,
)
from app.services.loaders import load_csv, load_email_threads
from app.services.ratelimit import (
    consume_quota,
    consume_upload,
    remaining_quota,
    remaining_uploads,
)
from app.services.retriever import invalidate_caches
from app.services.seed import ensure_seeded
from app.services.suggestions import get_suggestions

router = APIRouter()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_owner(x_api_key: Optional[str]) -> bool:
    return bool(settings.api_key) and x_api_key == settings.api_key


@router.get("/api/suggestions")
async def suggestions():
    ensure_seeded()
    return {"suggestions": get_suggestions()}


@router.get("/api/limits")
async def get_limits(request: Request):
    ip = _client_ip(request)
    return {
        "free_remaining": remaining_quota(ip),
        "free_per_day": settings.free_queries_per_day,
        "uploads_remaining": remaining_uploads(ip),
        "uploads_per_day": settings.free_uploads_per_day,
        "provider": settings.llm_provider,
    }


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: Request,
    question: str = Form(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    owner = _is_owner(x_api_key)
    free_remaining: Optional[int] = None
    if not owner:
        allowed, free_remaining = consume_quota(_client_ip(request))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Daily question limit reached for this visitor. Try again later.",
            )

    ensure_seeded()

    started = time.perf_counter()
    result = qa_graph.invoke(
        {
            "question": question,
            "retrieved_chunks": [],
            "answer": "",
            "citations": [],
            "error": None,
            "matched_entity": None,
            "gen_ms": None,
        }
    )
    total_ms = int((time.perf_counter() - started) * 1000)

    if result.get("error"):
        # 422 for "can't answer / no docs"; the message is safe to surface.
        raise HTTPException(status_code=422, detail=result["error"])

    provider = get_provider()
    try:
        total_vectors = get_collection().count()
    except Exception:
        total_vectors = None

    return QueryResponse(
        answer=result["answer"],
        citations=[
            CitationResponse(
                source=c["source"],
                doc_type=c["doc_type"],
                record_id=c["record_id"],
                company_name=c["company_name"],
                text=c["text"],
            )
            for c in result["citations"]
        ],
        provider=provider.name,
        model=provider.model,
        chunks_used=len(result["retrieved_chunks"]),
        matched_entity=result.get("matched_entity"),
        gen_ms=result.get("gen_ms"),
        total_ms=total_ms,
        free_remaining=free_remaining,
        total_vectors=total_vectors,
        retrieved=[
            RetrievedRef(
                source=c["source"],
                doc_type=c["doc_type"],
                record_id=c["record_id"],
                company_name=c["company_name"],
                via=c["via"],
                score=round(c["score"], 4),
                text=c["text"][:200],
            )
            for c in result["retrieved_chunks"]
        ],
    )


@router.get("/collection/stats", response_model=CollectionStatsResponse)
async def collection_stats():
    ensure_seeded()
    collection = get_collection()
    count = collection.count()

    sources: list[str] = []
    doc_types: dict = {}
    if count > 0:
        all_meta = collection.get(include=["metadatas"])
        metas = all_meta["metadatas"]
        sources = sorted({m.get("source", "") for m in metas if m.get("source")})
        doc_types = dict(Counter(m.get("doc_type", "") for m in metas))

    return CollectionStatsResponse(total_chunks=count, sources=sources, doc_types=doc_types)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    owner = _is_owner(x_api_key)
    file_bytes = await file.read()

    limit_mb = settings.max_file_size_mb if owner else settings.max_upload_mb
    if len(file_bytes) > limit_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {limit_mb} MB limit")

    name = (file.filename or "document").lower()
    kind = filetype.guess(file_bytes)
    mime = kind.mime if kind else None

    if not owner:
        allowed, _ = consume_upload(_client_ip(request))
        if not allowed:
            raise HTTPException(status_code=429, detail="Daily upload limit reached.")

    try:
        if mime == "application/pdf" or name.endswith(".pdf"):
            max_pages = None if owner else settings.max_demo_pdf_pages
            result = ingest_pdf(file_bytes, file.filename or "document.pdf", max_pages=max_pages)
        else:
            text = file_bytes.decode("utf-8-sig", errors="replace")
            if name.endswith(".csv"):
                res = ingest_documents(load_csv(text, file.filename or "upload.csv"))
                result = {"filename": file.filename, "chunks_added": res["chunks_added"]}
            elif name.endswith(".json"):
                res = ingest_documents(load_email_threads(text, file.filename or "upload.json"))
                result = {"filename": file.filename, "chunks_added": res["chunks_added"]}
            else:
                result = ingest_text(text, file.filename or "document.txt")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}")

    invalidate_caches()
    return IngestResponse(
        filename=result["filename"],
        chunks_added=result["chunks_added"],
        message=f"Ingested {result['chunks_added']} chunks from {result['filename']}",
    )
