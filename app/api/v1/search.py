from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.deps import LLM, CurrentUser, DbSession, get_embedder, get_embedding_cache
from app.core.cache import EmbeddingCache
from app.db.models import Role
from app.rag.embeddings import EmbeddingProvider
from app.schemas.search import (
    AnswerResponse,
    SearchHistoryItem,
    SearchRequest,
    SearchResponse,
)
from app.services.answer_service import AnswerService
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])

Embedder = Annotated[EmbeddingProvider, Depends(get_embedder)]
Cache = Annotated[EmbeddingCache, Depends(get_embedding_cache)]


class SearchHistoryResponse(BaseModel):
    items: list[SearchHistoryItem]
    total: int
    page: int
    page_size: int


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    user: CurrentUser,
    session: DbSession,
    embedder: Embedder,
    cache: Cache,
) -> SearchResponse:
    """Semantic search across indexed policy documents."""
    return await SearchService(session, embedder=embedder, cache=cache).search(
        user_id=user.id, query=payload.query, top_k=payload.top_k
    )


@router.post("/answer", response_model=AnswerResponse)
async def answer(
    payload: SearchRequest,
    user: CurrentUser,
    session: DbSession,
    embedder: Embedder,
    cache: Cache,
    llm: LLM,
) -> AnswerResponse:
    """Answer a policy question from retrieved passages, with citations."""
    return await AnswerService(
        session, embedder=embedder, cache=cache, llm=llm
    ).answer(user_id=user.id, query=payload.query, top_k=payload.top_k)


@router.get("/history", response_model=SearchHistoryResponse)
async def search_history(
    user: CurrentUser,
    session: DbSession,
    embedder: Embedder,
    cache: Cache,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> SearchHistoryResponse:
    """Query history. Administrators see every user's searches."""
    scope_user = None if user.role is Role.ADMIN else user.id
    items, total = await SearchService(session, embedder=embedder, cache=cache).history(
        user_id=scope_user, page=page, page_size=page_size
    )
    return SearchHistoryResponse(items=items, total=total, page=page, page_size=page_size)
