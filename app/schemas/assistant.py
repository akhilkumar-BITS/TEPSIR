import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import MessageRole
from app.schemas.search import Citation


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    ok: bool
    summary: str


class ConversationMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    tool_calls: list[ToolCallRecord] | None
    citations: list[Citation] | None
    model: str | None
    warning: str | None
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationSummaryOut(ConversationOut):
    message_count: int


class ConversationDetailOut(ConversationOut):
    messages: list[ConversationMessageOut]


class ConversationListOut(BaseModel):
    items: list[ConversationSummaryOut]
    total: int
    page: int
    page_size: int


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class MessageCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
