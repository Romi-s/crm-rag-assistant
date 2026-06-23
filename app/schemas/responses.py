from typing import List, Optional

from pydantic import BaseModel


class CitationResponse(BaseModel):
    source: str
    doc_type: str
    record_id: str = ""
    company_name: str = ""
    text: str


class RetrievedRef(BaseModel):
    source: str
    doc_type: str
    record_id: str = ""
    company_name: str = ""
    via: str = "hybrid"
    score: float = 0.0
    text: str = ""


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]
    provider: str                 # "local" / "bedrock"
    model: str
    chunks_used: int
    matched_entity: Optional[str] = None
    gen_ms: Optional[int] = None
    total_ms: Optional[int] = None
    free_remaining: Optional[int] = None
    total_vectors: Optional[int] = None
    retrieved: List[RetrievedRef] = []


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


class CollectionStatsResponse(BaseModel):
    total_chunks: int
    sources: List[str]
    doc_types: dict = {}
