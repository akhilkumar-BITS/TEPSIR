"""Upload, index, list and delete policy documents."""

import asyncio
import hashlib
import logging
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    ConflictError,
    IndexingError,
    NotFoundError,
    UnsupportedDocumentError,
    ValidationError,
)
from app.db.models import Document
from app.rag.chunking import chunk_pages
from app.rag.embeddings import EmbeddingProvider
from app.rag.extract import SUPPORTED_EXTENSIONS, extract_pages
from app.rag.vector_store import PgVectorStore
from app.repositories.document_repository import DocumentRepository
from app.storage.base import ObjectStore

logger = logging.getLogger(__name__)

# Content types we are willing to attach to an uploaded object.
_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        store: ObjectStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self.session = session
        self.store = store
        self.embedder = embedder
        self.documents = DocumentRepository(session)
        self.vectors = PgVectorStore(session)

    async def upload(
        self,
        *,
        title: str,
        description: str | None,
        filename: str,
        content_type: str | None,
        data: bytes,
        uploaded_by: uuid.UUID,
    ) -> Document:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedDocumentError(
                f"Unsupported file type '{suffix or filename}'. "
                f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )
        if not data:
            raise ValidationError("The uploaded file is empty.")
        if len(data) > settings.max_upload_bytes:
            raise ValidationError(
                f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit."
            )
        if not title.strip():
            raise ValidationError("A title is required.")

        checksum = hashlib.sha256(data).hexdigest()
        duplicate = await self.documents.get_by_checksum(checksum)
        if duplicate is not None:
            raise ConflictError(
                f"This file was already uploaded as '{duplicate.title}'."
            )

        resolved_type = _CONTENT_TYPES[suffix]
        storage_key = f"documents/{checksum}{suffix}"
        await self.store.put(storage_key, data, resolved_type)

        document = await self.documents.create(
            title=title.strip(),
            description=(description or "").strip() or None,
            original_filename=filename,
            content_type=resolved_type,
            storage_key=storage_key,
            size_bytes=len(data),
            checksum_sha256=checksum,
            uploaded_by=uploaded_by,
        )

        try:
            chunk_count = await self._index(document, data)
        except Exception as exc:
            logger.exception("Indexing failed for document %s", document.id)
            await self.documents.mark_failed(document, str(exc) or exc.__class__.__name__)
            await self.session.commit()
            raise IndexingError(
                "The document was stored but could not be indexed. "
                "Check the document status for details."
            ) from exc

        await self.documents.mark_indexed(document, chunk_count)
        await self.session.commit()
        await self.session.refresh(document)
        logger.info("Indexed document %s (%d chunks)", document.id, chunk_count)
        return document

    async def _index(self, document: Document, data: bytes) -> int:
        pages = await asyncio.to_thread(
            extract_pages, document.original_filename, document.content_type, data
        )
        if not pages:
            raise IndexingError("No extractable text was found in the document.")

        chunks = chunk_pages(
            pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        if not chunks:
            raise IndexingError("The document produced no text chunks.")

        embeddings = await self.embedder.embed_documents([chunk.content for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise IndexingError("Embedding provider returned an unexpected number of vectors.")

        # Replace any stale chunks when re-indexing the same document.
        await self.vectors.delete_document(document.id)
        indexed = [
            _to_indexed(chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        return await self.vectors.add_chunks(document_id=document.id, chunks=indexed)

    async def list_documents(self, *, page: int, page_size: int) -> tuple[list[Document], int]:
        if page < 1 or page_size < 1 or page_size > 100:
            raise ValidationError("page must be >= 1 and page_size between 1 and 100.")
        return await self.documents.list_documents(offset=(page - 1) * page_size, limit=page_size)

    async def get_document(self, document_id: uuid.UUID) -> Document:
        document = await self.documents.get(document_id)
        if document is None:
            raise NotFoundError("Document not found.")
        return document

    async def get_chunks(self, document_id: uuid.UUID):
        await self.get_document(document_id)
        return await self.documents.list_chunks(document_id)

    async def delete_document(self, document_id: uuid.UUID) -> None:
        document = await self.get_document(document_id)
        await self.documents.delete(document)
        await self.session.commit()
        try:
            await self.store.delete(document.storage_key)
        except Exception:  # noqa: BLE001 - orphaned objects are not worth failing the request
            logger.warning("Could not delete stored object %s", document.storage_key)


def _to_indexed(chunk, embedding):
    from app.rag.vector_store import IndexedChunk

    return IndexedChunk(chunk=chunk, embedding=embedding)
