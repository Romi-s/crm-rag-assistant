"""Chunk -> embed (local) -> upsert into ChromaDB.

Structured records (`doc.split is False`) are stored as a single chunk so a
customer/ticket/email stays one coherent, citable unit. Prose (`doc.split is True`)
is split with a recursive character splitter. Every chunk keeps the document's
`doc_type` / `customer_id` / `company_name` metadata for citations and entity-aware
retrieval.
"""

import hashlib
from typing import List, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.embeddings import embed_texts
from app.services.loaders import (
    Document,
    load_dataset,
    load_pdf,
    load_text,
)


def get_chroma_client() -> chromadb.ClientAPI:
    if settings.chroma_host:
        kwargs = {
            "host": settings.chroma_host,
            "port": settings.chroma_port,
            "ssl": settings.chroma_ssl,
        }
        if settings.chroma_token:
            kwargs["headers"] = {"Authorization": f"Bearer {settings.chroma_token}"}
        return chromadb.HttpClient(**kwargs)
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection() -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Drop and recreate the collection (used by a full --rebuild)."""
    client = get_chroma_client()
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    client.get_or_create_collection(
        name=settings.collection_name, metadata={"hnsw:space": "cosine"}
    )
    _invalidate_caches()


def _invalidate_caches() -> None:
    # Imported lazily to avoid a circular import (retriever depends on this module).
    from app.services.retriever import invalidate_caches

    invalidate_caches()


def ingest_documents(docs: List[Document], rebuild: bool = False) -> dict:
    if rebuild:
        reset_collection()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for doc in docs:
        pieces = splitter.split_text(doc.text) if doc.split else [doc.text]
        for i, piece in enumerate(pieces):
            if not piece.strip():
                continue
            meta = doc.metadata()
            meta["chunk_index"] = i
            chunk_id = hashlib.sha256(
                f"{doc.source}:{doc.doc_type}:{doc.record_id}:{i}:{piece[:50]}".encode()
            ).hexdigest()[:16]
            texts.append(piece)
            metadatas.append(meta)
            ids.append(chunk_id)

    if not texts:
        return {"documents": len(docs), "chunks_added": 0}

    embeddings = embed_texts(texts)
    collection = get_collection()

    batch_size = 2000
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    _invalidate_caches()
    return {"documents": len(docs), "chunks_added": len(texts)}


def ingest_dataset(root: Optional[str] = None, rebuild: bool = True) -> dict:
    """Load and index the whole dataset folder."""
    docs = load_dataset(root or settings.dataset_dir)
    result = ingest_documents(docs, rebuild=rebuild)
    result["source_files"] = len({d.source for d in docs})
    return result


# --- single-file upload helpers (used by the /ingest endpoint) --------------- #
def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    max_pages: Optional[int] = None,
) -> dict:
    docs = load_pdf(pdf_bytes, filename, max_pages=max_pages)
    res = ingest_documents(docs)
    return {"filename": filename, "chunks_added": res["chunks_added"]}


def ingest_text(text: str, filename: str) -> dict:
    docs = load_text(text, filename)
    res = ingest_documents(docs)
    return {"filename": filename, "chunks_added": res["chunks_added"]}
