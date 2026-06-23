import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.services.seed import ensure_seeded

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("crm-rag")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting CRM RAG Assistant | provider=%s | model=%s | embed=%s",
             settings.llm_provider, settings.ollama_model, settings.embedding_model)
    # Index the dataset in the background so startup / health stay fast while the
    # embedding model downloads and the corpus is embedded.
    threading.Thread(target=ensure_seeded, daemon=True).start()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="CRM RAG Assistant",
        description=(
            "Local-first Retrieval-Augmented Generation over internal CRM data — "
            "hybrid + entity-aware retrieval, grounded answers with source citations, "
            "pluggable local (Ollama) / Bedrock generation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "provider": settings.llm_provider}

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
        )

    # UI mounted last so API routes take precedence; html=True serves index.html at "/".
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


app = create_app()
