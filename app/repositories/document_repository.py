from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Document, DocumentChunk, DocumentStatus, SearchQuery


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        title: str,
        description: str | None,
        original_filename: str,
        content_type: str,
        storage_key: str,
        size_bytes: int,
        checksum_sha256: str,
        uploaded_by: uuid.UUID,
    ) -> Document:
        document = Document(
            title=title,
            description=description,
            original_filename=original_filename,
            content_type=content_type,
            storage_key=storage_key,
            size_bytes=size_bytes,
            checksum_sha256=checksum_sha256,
            uploaded_by=uploaded_by,
            status=DocumentStatus.PROCESSING,
        )
        self.session.add(document)
        await self.session.flush()
        return document

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def get_by_checksum(self, checksum: str) -> Document | None:
        return await self.session.scalar(
            select(Document).where(Document.checksum_sha256 == checksum)
        )

    async def list_documents(self, *, offset: int, limit: int) -> tuple[list[Document], int]:
        total = await self.session.scalar(select(func.count()).select_from(Document)) or 0
        stmt = (
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list((await self.session.scalars(stmt)).all()), total

    async def list_chunks(self, document_id: uuid.UUID) -> Sequence[DocumentChunk]:
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return (await self.session.scalars(stmt)).all()

    async def mark_indexed(self, document: Document, chunk_count: int) -> None:
        document.status = DocumentStatus.INDEXED
        document.chunk_count = chunk_count
        document.error_message = None
        await self.session.flush()

    async def mark_failed(self, document: Document, error_message: str) -> None:
        document.status = DocumentStatus.FAILED
        document.error_message = error_message[:2000]
        await self.session.flush()

    async def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        self.session.add_all(chunks)
        await self.session.flush()

    async def delete_chunks(self, document_id: uuid.UUID) -> None:
        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )

    async def delete(self, document: Document) -> None:
        await self.session.delete(document)
        await self.session.flush()

    # -- retrieval ---------------------------------------------------------

    async def search_chunks(
        self, *, embedding: list[float], top_k: int
    ) -> list[tuple[DocumentChunk, Document, float]]:
        """Nearest chunks by cosine distance, joined with their document."""
        distance = DocumentChunk.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(DocumentChunk, Document, distance)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == DocumentStatus.INDEXED)
            .order_by(distance)
            .limit(top_k)
            .options(selectinload(DocumentChunk.document))
        )
        rows = (await self.session.execute(stmt)).all()
        # pgvector returns distance in [0, 2]; map to similarity in [0, 1].
        return [(chunk, document, 1.0 - float(dist)) for chunk, document, dist in rows]

    # -- search audit ------------------------------------------------------

    async def log_search(
        self, *, user_id: uuid.UUID, query: str, top_k: int, result_count: int
    ) -> SearchQuery:
        record = SearchQuery(
            user_id=user_id, query=query, top_k=top_k, result_count=result_count
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def list_searches(
        self, *, user_id: uuid.UUID | None, offset: int, limit: int
    ) -> tuple[list[SearchQuery], int]:
        conditions = [] if user_id is None else [SearchQuery.user_id == user_id]
        total = (
            await self.session.scalar(
                select(func.count()).select_from(SearchQuery).where(*conditions)
            )
            or 0
        )
        stmt = (
            select(SearchQuery)
            .where(*conditions)
            .order_by(SearchQuery.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list((await self.session.scalars(stmt)).all()), total
