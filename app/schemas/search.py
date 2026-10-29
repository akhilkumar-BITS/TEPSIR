import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    page_number: int | None
    content: str
    # Cosine similarity in [0, 1]; higher is more relevant.
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    elapsed_ms: int


class Citation(BaseModel):
    # 1-based passage number used by the answer, so clients can highlight it.
    marker: int = Field(ge=1)
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page_number: int | None
    score: float


class AnswerResponse(BaseModel):
    query: str
    # None when synthesis is disabled, unavailable, or had nothing to work with.
    answer: str | None
    citations: list[Citation]
    results: list[SearchHit]
    model: str | None
    elapsed_ms: int
    warning: str | None


class SearchHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    query: str
    top_k: int
    result_count: int
    created_at: datetime
