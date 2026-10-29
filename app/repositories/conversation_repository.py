from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, ConversationMessage, MessageRole


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, *, user_id: uuid.UUID, title: str) -> Conversation:
        conversation = Conversation(user_id=user_id, title=title)
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def get(self, conversation_id: uuid.UUID, *, user_id: uuid.UUID) -> Conversation | None:
        """Load a conversation only if it belongs to ``user_id``."""
        return await self.session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.user_id == user_id
            )
        )

    async def list_for_user(
        self, *, user_id: uuid.UUID, offset: int, limit: int
    ) -> tuple[list[tuple[Conversation, int]], int]:
        total = (
            await self.session.scalar(
                select(func.count())
                .select_from(Conversation)
                .where(Conversation.user_id == user_id)
            )
            or 0
        )
        stmt = (
            select(Conversation, func.count(ConversationMessage.id))
            .outerjoin(
                ConversationMessage,
                ConversationMessage.conversation_id == Conversation.id,
            )
            .where(Conversation.user_id == user_id)
            .group_by(Conversation.id)
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(conversation, count) for conversation, count in rows], total

    async def messages(
        self, conversation_id: uuid.UUID, *, limit: int | None = None
    ) -> list[ConversationMessage]:
        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at, ConversationMessage.id)
        )
        if limit is None:
            return list((await self.session.scalars(stmt)).all())
        # The newest ``limit`` messages, returned oldest first.
        newest = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
            .limit(limit)
        )
        rows = list((await self.session.scalars(newest)).all())
        rows.reverse()
        return rows

    async def add_message(
        self,
        *,
        conversation_id: uuid.UUID,
        role: MessageRole,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        citations: list[dict[str, Any]] | None = None,
        model: str | None = None,
        warning: str | None = None,
    ) -> ConversationMessage:
        message = ConversationMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            citations=citations,
            model=model,
            warning=warning,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def rename(self, conversation: Conversation, title: str) -> None:
        conversation.title = title
        await self.session.flush()

    async def touch(self, conversation: Conversation) -> None:
        """Bump ``updated_at`` so the conversation list sorts by latest activity."""
        conversation.updated_at = datetime.now(UTC)
        await self.session.flush()

    async def delete(self, conversation: Conversation) -> None:
        await self.session.delete(conversation)
        await self.session.flush()

    async def commit(self) -> None:
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
