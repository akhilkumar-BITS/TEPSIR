import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import DocumentStatus


class DocumentPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    description: str | None
    original_filename: str
    content_type: str
    size_bytes: int
    checksum_sha256: str
    status: DocumentStatus
    chunk_count: int
    error_message: str | None
    uploaded_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentPublic]
    total: int
    page: int
    page_size: int


class DocumentChunkPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_index: int
    page_number: int | None
    content: str


class DocumentChunkList(BaseModel):
    document_id: uuid.UUID
    items: list[DocumentChunkPublic]
    total: int
