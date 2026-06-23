"""Retrieval = hybrid search  +  entity-aware gather.

Two complementary strategies, merged:

1. **Hybrid search** — semantic (Chroma cosine over local embeddings) + lexical
   (BM25), fused with Reciprocal Rank Fusion. Good general recall.

2. **Entity-aware gather** — if the question names a known customer/company (or a
   CUST-xxx id), we *force-include* that customer's records across every source
   (CRM row, sales notes, tickets, emails) via a metadata filter. This is what makes
   "summarise customer X" and "open critical tickets for X" reliable instead of
   hoping similarity search happens to surface all the relevant rows.

Both BM25 and the entity index are cached and rebuilt lazily after any ingest
(`invalidate_caches`).
"""

import re
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.agent.state import RetrievedChunk
from app.config import settings
from app.services.embeddings import embed_query
from app.services.ingest import get_collection

_bm25_cache: Optional[tuple] = None
_entity_cache: Optional[tuple] = None  # (companies: dict, id_to_company: dict)

_COMPANY_SUFFIXES = {
    "llc", "inc", "ltd", "limited", "co", "corp", "corporation", "group",
    "plc", "llp", "gmbh", "sa", "ag", "srl", "bv", "pvt", "private", "holding",
    "holdings", "company", "trading", "industries", "solutions", "services",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


# --------------------------------------------------------------------------- #
# cache management
# --------------------------------------------------------------------------- #
def invalidate_caches() -> None:
    global _bm25_cache, _entity_cache
    _bm25_cache = None
    _entity_cache = None


def _build_bm25_index() -> None:
    global _bm25_cache
    collection = get_collection()
    all_docs = collection.get(include=["documents", "metadatas"])
    if not all_docs["documents"]:
        _bm25_cache = None
        return
    tokenized = [_tokenize(doc) for doc in all_docs["documents"]]
    _bm25_cache = (
        BM25Okapi(tokenized),
        all_docs["ids"],
        all_docs["documents"],
        all_docs["metadatas"],
    )


def _company_core(name: str) -> str:
    """Strip legal/common suffixes so 'Alpha Trading LLC' matches 'Alpha Trading'."""
    tokens = [t for t in re.findall(r"\w+", name.lower())]
    core = [t for t in tokens if t not in _COMPANY_SUFFIXES]
    return " ".join(core or tokens)


def _build_entity_index() -> None:
    global _entity_cache
    collection = get_collection()
    meta = collection.get(include=["metadatas"])
    companies: dict[str, Tuple[str, str]] = {}   # core -> (company_name, customer_id)
    id_to_company: dict[str, str] = {}
    for m in meta.get("metadatas") or []:
        comp = (m or {}).get("company_name") or ""
        cid = (m or {}).get("customer_id") or ""
        if comp:
            core = _company_core(comp)
            # keep the longest core for a given company spelling
            if core and core not in companies:
                companies[core] = (comp, cid)
            if cid:
                id_to_company.setdefault(cid, comp)
    _entity_cache = (companies, id_to_company)


def _detect_entity(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (customer_id, company_name) if the query names a known entity."""
    global _entity_cache
    if _entity_cache is None:
        _build_entity_index()
    if not _entity_cache:
        return None, None
    companies, id_to_company = _entity_cache
    q = query.lower()

    m = re.search(r"\bcust-\d+\b", q)
    if m:
        cid = m.group(0).upper()
        return cid, id_to_company.get(cid)

    best: Optional[Tuple[str, str, str]] = None  # (core, company, customer_id)
    for core, (comp, cid) in companies.items():
        if len(core) < 3:
            continue
        if re.search(r"\b" + re.escape(core) + r"\b", q):
            if best is None or len(core) > len(best[0]):
                best = (core, comp, cid)
    if best:
        return (best[2] or None), best[1]
    return None, None


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #
def _to_chunk(doc_id: str, text: str, meta: dict, score: float, via: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=doc_id,
        text=text,
        source=meta.get("source", ""),
        doc_type=meta.get("doc_type", ""),
        record_id=meta.get("record_id", ""),
        customer_id=meta.get("customer_id", ""),
        company_name=meta.get("company_name", ""),
        chunk_index=int(meta.get("chunk_index", 0) or 0),
        score=score,
        via=via,
    )


def hybrid_retrieve(query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    if top_k is None:
        top_k = settings.final_top_k
    retrieval_k = settings.retrieval_top_k

    collection = get_collection()
    doc_count = collection.count()
    if doc_count == 0:
        return []

    # --- vector search ---
    query_embedding = embed_query(query)
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(retrieval_k, doc_count),
        include=["documents", "metadatas", "distances"],
    )

    # --- BM25 search ---
    global _bm25_cache
    if _bm25_cache is None:
        _build_bm25_index()

    bm25_ranked: list[dict] = []
    if _bm25_cache is not None:
        bm25, ids, docs, metas = _bm25_cache
        scores = bm25.get_scores(_tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :retrieval_k
        ]
        bm25_ranked = [
            {"id": ids[i], "document": docs[i], "metadata": metas[i]}
            for i in top_indices
            if scores[i] > 0
        ]

    # --- Reciprocal Rank Fusion ---
    K = 60
    rrf: dict[str, float] = {}
    lookup: dict[str, dict] = {}

    if vector_results["ids"] and vector_results["ids"][0]:
        for rank, doc_id in enumerate(vector_results["ids"][0]):
            rrf[doc_id] = rrf.get(doc_id, 0) + 1 / (K + rank + 1)
            lookup[doc_id] = {
                "document": vector_results["documents"][0][rank],
                "metadata": vector_results["metadatas"][0][rank],
            }

    for rank, item in enumerate(bm25_ranked):
        doc_id = item["id"]
        rrf[doc_id] = rrf.get(doc_id, 0) + 1 / (K + rank + 1)
        lookup.setdefault(doc_id, {"document": item["document"], "metadata": item["metadata"]})

    sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k]
    return [
        _to_chunk(i, lookup[i]["document"], lookup[i]["metadata"], rrf[i], "hybrid")
        for i in sorted_ids
    ]


def entity_retrieve(customer_id: Optional[str], company: Optional[str]) -> List[RetrievedChunk]:
    """Force-include all records for a named customer/company, across sources."""
    collection = get_collection()
    found: dict[str, RetrievedChunk] = {}

    filters = []
    if customer_id:
        filters.append({"customer_id": customer_id})
    if company:
        filters.append({"company_name": company})

    for where in filters:
        try:
            res = collection.get(where=where, include=["documents", "metadatas"])
        except Exception:
            continue
        for doc_id, text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            found.setdefault(doc_id, _to_chunk(doc_id, text, meta, 1.0, "entity"))

    # Spread doc_types: CRM row/lead first, then notes/tickets/emails, so a capped
    # context still shows a cross-section rather than (say) 8 tickets.
    priority = {"customer": 0, "lead": 1, "sales_note": 2, "ticket": 3, "email": 4}
    chunks = sorted(found.values(), key=lambda c: priority.get(c["doc_type"], 9))
    return chunks[: settings.entity_top_k]


def retrieve(query: str, top_k: Optional[int] = None) -> Tuple[List[RetrievedChunk], Optional[str]]:
    """Top-level retrieval: entity records (if any) force-included, then hybrid.

    Returns (chunks, matched_company_name)."""
    if top_k is None:
        top_k = settings.final_top_k

    customer_id, company = _detect_entity(query)
    entity_chunks = entity_retrieve(customer_id, company) if (customer_id or company) else []
    entity_ids = {c["id"] for c in entity_chunks}

    hybrid = [c for c in hybrid_retrieve(query, top_k=top_k) if c["id"] not in entity_ids]

    merged = entity_chunks + hybrid[:top_k]
    return merged, company
