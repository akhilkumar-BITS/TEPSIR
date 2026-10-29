"""Conversations and messages for the policy assistant.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    message_role = postgresql.ENUM("USER", "ASSISTANT", name="message_role", create_type=False)
    message_role.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_conversations_user_id_users",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])

    op.create_table(
        "conversation_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", message_role, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("warning", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_conversation_messages_conversation_id_conversations",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_conversation_messages_conversation_id",
        "conversation_messages",
        ["conversation_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_messages_conversation_id", table_name="conversation_messages"
    )
    op.drop_table("conversation_messages")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_table("conversations")

    op.execute("DROP TYPE IF EXISTS message_role")
