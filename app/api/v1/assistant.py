"""Conversational policy assistant endpoints.

The message endpoint streams Server-Sent Events. Everything that can fail with a
meaningful HTTP status (unknown conversation, no API key, empty message) happens
in ``prepare_turn`` before the response starts; once the stream is open, failures
are emitted as ``error`` frames because the status line can no longer change.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response, status
from sse_starlette.sse import EventSourceResponse

from app.api.deps import CurrentUser, DbSession, get_embedder, get_embedding_cache
from app.core.cache import EmbeddingCache
from app.core.config import settings
from app.rag.embeddings import EmbeddingProvider
from app.schemas.assistant import (
    ConversationCreateRequest,
    ConversationDetailOut,
    ConversationListOut,
    ConversationMessageOut,
    ConversationOut,
    ConversationSummaryOut,
    MessageCreateRequest,
)
from app.services.assistant_service import AssistantService

router = APIRouter(prefix="/assistant", tags=["assistant"])

Embedder = Annotated[EmbeddingProvider, Depends(get_embedder)]
Cache = Annotated[EmbeddingCache, Depends(get_embedding_cache)]


def _service(
    session: DbSession, user: CurrentUser, embedder: Embedder, cache: Cache
) -> AssistantService:
    return AssistantService(session, user_id=user.id, embedder=embedder, cache=cache)


@router.post(
    "/conversations",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    payload: ConversationCreateRequest,
    user: CurrentUser,
    session: DbSession,
    embedder: Embedder,
    cache: Cache,
) -> ConversationOut:
    """Start a conversation. The title is derived from the first message if omitted."""
    conversation = await _service(session, user, embedder, cache).create(title=payload.title)
    return ConversationOut.model_validate(conversation)


@router.get("/conversations", response_model=ConversationListOut)
async def list_conversations(
    user: CurrentUser,
    session: DbSession,
    embedder: Embedder,
    cache: Cache,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=50),
) -> ConversationListOut:
    """The caller's conversations, most recently active first."""
    rows, total = await _service(session, user, embedder, cache).list(
        page=page, page_size=page_size
    )
    items = [
        ConversationSummaryOut(
            **ConversationOut.model_validate(conversation).model_dump(),
            message_count=message_count,
        )
        for conversation, message_count in rows
    ]
    return ConversationListOut(items=items, total=total, page=page, page_size=page_size)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(
    conversation_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    embedder: Embedder,
    cache: Cache,
) -> ConversationDetailOut:
    """A conversation with its full message history, including tool traces."""
    conversation, messages = await _service(session, user, embedder, cache).detail(
        conversation_id
    )
    return ConversationDetailOut(
        **ConversationOut.model_validate(conversation).model_dump(),
        messages=[ConversationMessageOut.model_validate(message) for message in messages],
    )


@router.delete(
    "/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    embedder: Embedder,
    cache: Cache,
) -> Response:
    await _service(session, user, embedder, cache).delete(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: uuid.UUID,
    payload: MessageCreateRequest,
    user: CurrentUser,
    session: DbSession,
    embedder: Embedder,
    cache: Cache,
) -> EventSourceResponse:
    """Send a message and stream the assistant's turn as Server-Sent Events.

    Events: ``start``, ``token``, ``tool_call``, ``tool_result``, ``status``,
    ``answer_reset`` (the reviewer rejected the draft), ``citations``, ``done``,
    ``error``. Event data is JSON.
    """
    service = _service(session, user, embedder, cache)
    turn = await service.prepare_turn(conversation_id=conversation_id, content=payload.content)

    async def events() -> Any:
        async for name, data in service.stream_turn(turn):
            yield {
                "event": name,
                "data": json.dumps(data, ensure_ascii=False, default=str),
            }

    return EventSourceResponse(
        events(),
        ping=settings.agent_heartbeat_seconds,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
