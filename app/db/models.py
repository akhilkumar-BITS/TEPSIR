"""Persistence models.

The schema separates three concerns:
  * identity       - ``users`` (role decides what a caller may do)
  * corpus         - ``documents`` (metadata) and ``document_chunks`` (retrieval units)
  * observability  - ``search_queries`` (what was asked and what came back)

Embeddings live next to the chunk text. Keeping them in PostgreSQL (pgvector)
avoids a second datastore at this scale and keeps deletes transactional.
"""

import enum
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Enum, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Role(enum.StrEnum):
    """Roles are stored as their wire names to keep JWTs and DB rows identical."""

    ADMIN = "ROLE_ADMIN"
    USER = "ROLE_USER"


class DocumentStatus(enum.StrEnum):
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


class MessageRole(enum.StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Role] = mapped_column(
        Enum(Role, name="user_role", values_callable=lambda e: [m.value for m in e]),
        default=Role.USER,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    documents: Mapped[list["Document"]] = relationship(
        back_populates="uploaded_by_user", cascade="all, delete-orphan"
    )
    searches: Mapped[list["SearchQuery"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"

    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    content_type: Mapped[str] = mapped_column(String(150), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            name="document_status",
            values_callable=lambda e: [m.value for m in e],
        ),
        default=DocumentStatus.PROCESSING,
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    uploaded_by_user: Mapped[User] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentChunk(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_id"),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimension), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class SearchQuery(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "search_queries"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query: Mapped[str] = mapped_column(String(1000), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False)

    user: Mapped[User] = relationship(back_populates="searches")


class Conversation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "conversations"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ConversationMessage.created_at",
    )


class ConversationMessage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "conversation_messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, name="message_role", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Tool trace and validated citations are stored with the answer so a reload
    # reproduces exactly what the agent did, without re-running it.
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    warning: Mapped[str | None] = mapped_column(Text, nullable=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
