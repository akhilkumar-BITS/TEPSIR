"""Vector store abstraction.

``PgVectorStore`` stores chunks in ``document_chunks`` and uses pgvector's
cosine distance operator. The protocol exists so a dedicated store (Qdrant,
Weaviate, ...) can be swapped in without touching the search service.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk
from app.rag.chunking import TextChunk
from app.repositories.document_repository import DocumentRepository


@dataclass(slots=True)
class IndexedChunk:
    chunk: TextChunk
    embedding: list[float]


@dataclass(slots=True)
class ChunkMatch:
    chunk: DocumentChunk
    document: Document
    score: float


class VectorStore(Protocol):
    async def add_chunks(
        self, *, document_id: uuid.UUID, chunks: Sequence[IndexedChunk]
    ) -> int: ...

    async def delete_document(self, document_id: uuid.UUID) -> None: ...

    async def search(self, *, embedding: list[float], top_k: int) -> list[ChunkMatch]: ...


class PgVectorStore:
    def __init__(self, session: AsyncSession) -> None:
        self.repository = DocumentRepository(session)

    async def add_chunks(
        self, *, document_id: uuid.UUID, chunks: Sequence[IndexedChunk]
    ) -> int:
        records = [
            DocumentChunk(
                document_id=document_id,
                chunk_index=item.chunk.index,
                page_number=item.chunk.page_number,
                content=item.chunk.content,
                embedding=item.embedding,
            )
            for item in chunks
        ]
        await self.repository.add_chunks(records)
        return len(records)

    async def delete_document(self, document_id: uuid.UUID) -> None:
        await self.repository.delete_chunks(document_id)

    async def search(self, *, embedding: list[float], top_k: int) -> list[ChunkMatch]:
        rows = await self.repository.search_chunks(embedding=embedding, top_k=top_k)
        return [
            ChunkMatch(chunk=chunk, document=document, score=score)
            for chunk, document, score in rows
        ]
