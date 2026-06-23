from typing import List, Optional, TypedDict


class RetrievedChunk(TypedDict):
    id: str
    text: str
    source: str
    doc_type: str
    record_id: str
    customer_id: str
    company_name: str
    chunk_index: int
    score: float
    via: str          # "hybrid" or "entity" — how this chunk was retrieved


class Citation(TypedDict):
    source: str
    doc_type: str
    record_id: str
    company_name: str
    text: str


class QAState(TypedDict):
    question: str
    retrieved_chunks: List[RetrievedChunk]
    answer: str
    citations: List[Citation]
    error: Optional[str]
    matched_entity: Optional[str]    # company we detected in the question, if any
    gen_ms: Optional[int]            # generation latency (observability)
