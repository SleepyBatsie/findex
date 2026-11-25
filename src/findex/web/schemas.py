"""Pydantic boundary schemas for FastAPI request validation and JSON responses."""

from typing import Literal

from pydantic import BaseModel, Field


class SearchHitOut(BaseModel):
    """Single paginated search hit returned by GET /search."""

    rank: int = Field(..., ge=1)
    doc_id: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    """Paginated response envelope for GET /search."""

    query: str
    scorer: Literal["bm25", "tfidf"]
    mode: Literal["lexical", "semantic", "hybrid"] = "lexical"
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1)
    total_hits: int = Field(..., ge=0)
    total_pages: int = Field(..., ge=0)
    has_prev: bool
    has_next: bool
    results: list[SearchHitOut]


class HealthResponse(BaseModel):
    """Service readiness and index status payload for GET /health."""

    status: Literal["ok", "unavailable"]
    index_loaded: bool
    semantic_loaded: bool = False
    num_docs: int = Field(..., ge=0)
    num_terms: int = Field(..., ge=0)


class DocumentOut(BaseModel):
    """Single document metadata and raw text returned by GET /docs/{doc_id}."""

    doc_id: str
    length: int = Field(..., ge=0)
    path: str | None = None
    text: str


class TermFreqOut(BaseModel):
    """Term frequency pair in corpus statistics."""

    term: str
    frequency: int = Field(..., ge=0)


class StatsResponse(BaseModel):
    """Index summary statistics payload for GET /stats."""

    num_docs: int = Field(..., ge=0)
    num_terms: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    avg_doc_length: float = Field(..., ge=0.0)
    top_terms: list[TermFreqOut]
