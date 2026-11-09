"""Semantic search over indexed policy chunks."""

import logging
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import EmbeddingCache
from app.core.config import settings
from app.core.exceptions import ValidationError
from app.rag.embeddings import EmbeddingProvider
from app.rag.reranking import Reranker
from app.rag.vector_store import ChunkMatch, PgVectorStore
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
        reranker: Reranker | None = None,
    ) -> None:
        self.session = session
        self.embedder = embedder
        self.cache = cache or EmbeddingCache()
        self.reranker = reranker
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

        # With a reranker, pull a wider candidate set and reorder it; otherwise
        # the vector-nearest top_k is the final answer.
        candidate_k = top_k
        if self.reranker is not None:
            candidate_k = min(
                max(top_k * settings.rerank_candidate_multiplier, top_k),
                settings.rerank_max_candidates,
            )
        matches = await self.vectors.search(embedding=embedding, top_k=candidate_k)
        logger.info("Found %s vectors", len(matches))
        if self.reranker is not None and matches:
            matches = await self._rerank(query, matches, top_k)
            logger.info("Found %s reranked matches", len(matches))
            
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
        logger.info(
            "Search by %s returned %d hits in %d ms (reranker=%s)",
            user_id,
            len(hits),
            elapsed_ms,
            self.reranker.name if self.reranker is not None else "off",
        )
        return SearchResponse(query=query, results=hits, elapsed_ms=elapsed_ms)

    async def _rerank(
        self, query: str, matches: list[ChunkMatch], top_k: int
    ) -> list[ChunkMatch]:
        """Reorder candidates with the cross-encoder, keeping the best top_k.

        Reranking is an optional accuracy boost; if the model cannot load or
        returns the wrong shape, log it and fall back to the vector order rather
        than failing the search.
        """
        assert self.reranker is not None  # narrowed by the caller
        try:
            scores = await self.reranker.score(query, [match.chunk.content for match in matches])
        except Exception:
            logger.exception("Reranking failed; falling back to vector order")
            return matches[:top_k]
        if len(scores) != len(matches):
            logger.warning(
                "Reranker returned %d scores for %d candidates; using vector order",
                len(scores),
                len(matches),
            )
            return matches[:top_k]
        # Cross-encoder logits squashed by sigmoid are tiny for most pairs, so an
        # absolute 0-1 score shows every card as "0% match". Rescale across the
        # candidate set instead: best = 1.0, worst = 0.0. The score then reads as
        # relative relevance for this query, not absolute model confidence.
        normalised = _min_max(scores)
        ranked = sorted(
            zip(matches, normalised, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [
            ChunkMatch(chunk=match.chunk, document=match.document, score=float(score))
            for match, score in ranked[:top_k]
        ]

    async def history(
        self, *, user_id: uuid.UUID | None, page: int, page_size: int
    ) -> tuple[list[SearchHistoryItem], int]:
        if page < 1 or page_size < 1 or page_size > 100:
            raise ValidationError("page must be >= 1 and page_size between 1 and 100.")
        rows, total = await self.documents.list_searches(
            user_id=user_id, offset=(page - 1) * page_size, limit=page_size
        )
        return [SearchHistoryItem.model_validate(row) for row in rows], total


def _min_max(scores: list[float]) -> list[float]:
    """Rescale scores to [0, 1] across the candidate set.

    A degenerate spread (all candidates scored alike) keeps the raw values, since
    normalising noise would manufacture a ranking that is not there.
    """
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-12:
        return [float(score) for score in scores]
    return [(float(score) - low) / (high - low) for score in scores]
