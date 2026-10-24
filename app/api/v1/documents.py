import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status

from app.api.deps import AdminUser, CurrentUser, DbSession, get_embedder, get_object_store
from app.core.config import settings
from app.rag.embeddings import EmbeddingProvider
from app.schemas.document import (
    DocumentChunkList,
    DocumentChunkPublic,
    DocumentList,
    DocumentPublic,
)
from app.services.document_service import DocumentService
from app.storage.base import ObjectStore

router = APIRouter(prefix="/documents", tags=["documents"])

Store = Annotated[ObjectStore, Depends(get_object_store)]
Embedder = Annotated[EmbeddingProvider, Depends(get_embedder)]


def _service(session, store: ObjectStore, embedder: EmbeddingProvider) -> DocumentService:
    return DocumentService(session, store=store, embedder=embedder)


@router.post("", response_model=DocumentPublic, status_code=status.HTTP_201_CREATED)
async def upload_document(
    admin: AdminUser,
    session: DbSession,
    store: Store,
    embedder: Embedder,
    file: Annotated[UploadFile, File(description="Policy document (PDF, DOCX, TXT, MD)")],
    title: Annotated[str, Form(min_length=2, max_length=300)],
    description: Annotated[str | None, Form()] = None,
) -> DocumentPublic:
    """Upload and index a policy document. Administrator only."""
    # Read one byte past the limit so oversized uploads are rejected without
    # buffering the whole file in memory.
    data = await file.read(settings.max_upload_bytes + 1)
    document = await _service(session, store, embedder).upload(
        title=title,
        description=description,
        filename=file.filename or "upload",
        content_type=file.content_type,
        data=data,
        uploaded_by=admin.id,
    )
    return DocumentPublic.model_validate(document)


@router.get("", response_model=DocumentList)
async def list_documents(
    _: CurrentUser,
    session: DbSession,
    store: Store,
    embedder: Embedder,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> DocumentList:
    items, total = await _service(session, store, embedder).list_documents(
        page=page, page_size=page_size
    )
    return DocumentList(
        items=[DocumentPublic.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentPublic)
async def get_document(
    document_id: uuid.UUID,
    _: CurrentUser,
    session: DbSession,
    store: Store,
    embedder: Embedder,
) -> DocumentPublic:
    document = await _service(session, store, embedder).get_document(document_id)
    return DocumentPublic.model_validate(document)


@router.get("/{document_id}/chunks", response_model=DocumentChunkList)
async def list_document_chunks(
    document_id: uuid.UUID,
    _: AdminUser,
    session: DbSession,
    store: Store,
    embedder: Embedder,
) -> DocumentChunkList:
    chunks = await _service(session, store, embedder).get_chunks(document_id)
    return DocumentChunkList(
        document_id=document_id,
        items=[DocumentChunkPublic.model_validate(chunk) for chunk in chunks],
        total=len(chunks),
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    _: AdminUser,
    session: DbSession,
    store: Store,
    embedder: Embedder,
) -> None:
    await _service(session, store, embedder).delete_document(document_id)
