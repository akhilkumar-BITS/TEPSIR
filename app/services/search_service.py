"""Semantic search over indexed policy chunks."""

import logging
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import EmbeddingCache
from app.core.config import settings
from app.core.exceptions import ValidationError
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import PgVectorStore
from app.repositories.document_repository import DocumentRepository
from app.schemas.search import SearchHistoryItem, SearchHit, SearchResponse

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        embedder: EmbeddingProvider,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.session = session
        self.embedder = embedder
        self.cache = cache or EmbeddingCache()
        self.vectors = PgVectorStore(session)
        self.documents = DocumentRepository(session)

    async def search(
        self, *, user_id: uuid.UUID, query: str, top_k: int
    ) -> SearchResponse:
        query = query.strip()
        if len(query) < 2:
            raise ValidationError("Enter at least two characters to search.")
        if len(query) > settings.max_search_query_length:
            raise ValidationError(
                f"Queries are limited to {settings.max_search_query_length} characters."
            )
        if not 1 <= top_k <= settings.search_top_k_max:
            raise ValidationError(f"top_k must be between 1 and {settings.search_top_k_max}.")

        started = time.perf_counter()

        embedding = await self.cache.get(query, self.embedder.name)
        if embedding is None:
            embedding = await self.embedder.embed_query(query)
            await self.cache.set(query, self.embedder.name, embedding)

        matches = await self.vectors.search(embedding=embedding, top_k=top_k)
        hits = [
            SearchHit(
                chunk_id=match.chunk.id,
                document_id=match.document.id,
                document_title=match.document.title,
                chunk_index=match.chunk.chunk_index,
                page_number=match.chunk.page_number,
                content=match.chunk.content,
                score=round(match.score, 4),
            )
            for match in matches
        ]

        await self.documents.log_search(
            user_id=user_id, query=query, top_k=top_k, result_count=len(hits)
        )
        await self.session.commit()

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Search by %s returned %d hits in %d ms", user_id, len(hits), elapsed_ms)
        return SearchResponse(query=query, results=hits, elapsed_ms=elapsed_ms)

    async def history(
        self, *, user_id: uuid.UUID | None, page: int, page_size: int
    ) -> tuple[list[SearchHistoryItem], int]:
        if page < 1 or page_size < 1 or page_size > 100:
            raise ValidationError("page must be >= 1 and page_size between 1 and 100.")
        rows, total = await self.documents.list_searches(
            user_id=user_id, offset=(page - 1) * page_size, limit=page_size
        )
        return [SearchHistoryItem.model_validate(row) for row in rows], total
