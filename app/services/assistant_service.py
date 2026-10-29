"""Conversation persistence and the streaming agent turn.

One turn is split in two: ``prepare_turn`` validates the request, checks that a
model is configured, stores the user message and builds the graph, so every
error that deserves a real HTTP status happens *before* the response starts.
``stream_turn`` then emits SSE events and stores the assistant message in a
``finally`` block, which means a client disconnect still keeps the partial
answer instead of losing the turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import ToolContext
from app.agent.graph import build_agent_graph
from app.agent.model import build_chat_model
from app.agent.tools import build_tools
from app.core.cache import EmbeddingCache
from app.core.config import settings
from app.core.exceptions import NotFoundError, ServiceUnavailableError, ValidationError
from app.db.models import Conversation, ConversationMessage, MessageRole
from app.db.session import SessionLocal
from app.rag.embeddings import EmbeddingProvider
from app.rag.prompting import extract_citation_markers
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New conversation"
_IDK_PREFIX = "i don't know based on the available policy passages"

_NOT_CONFIGURED = (
    "The policy assistant is not configured. Set DEEPSEEK_API_KEY and AGENT_ENABLED=true."
)
_TIMEOUT = "The assistant took too long and stopped before finishing."
_FAILED = "The assistant failed before finishing. Please try again."
_STEP_LIMIT = (
    "The assistant reached its tool-call limit before producing an answer. "
    "Try a narrower question."
)
_INVALID_CITATIONS = (
    "The answer referenced passages that were not retrieved; those citations were ignored."
)
_NO_CITATIONS = (
    "The answer did not cite any passage; verify it against the policy documents."
)

AgentEvent = tuple[str, dict[str, Any]]


@dataclass(slots=True)
class PreparedTurn:
    """Everything a stream needs, with no ORM object tied to the request session.

    The stream opens its own session because FastAPI closes request-scoped
    dependencies around the response, and a turn can outlive the request.
    """

    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    title: str
    history: list[BaseMessage]
    chat_model: BaseChatModel
    verifier_model: BaseChatModel
    model_name: str


def _message_text(message: BaseMessage | Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return ""


def _tool_failed(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    stripped = content.lstrip()
    # ToolNode converts a raised exception into an "Error: ..." string when
    # handle_tool_errors=True; the tools' own failures are JSON with "error".
    if stripped.startswith("Error:") or stripped.startswith("ToolException"):
        return True
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and "error" in payload


def _derive_title(content: str) -> str:
    title = " ".join(content.split())
    if len(title) > 80:
        title = title[:80].rstrip() + "…"
    return title or DEFAULT_TITLE


class AssistantService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        user_id: uuid.UUID,
        embedder: EmbeddingProvider,
        cache: EmbeddingCache | None = None,
        chat_model: BaseChatModel | None = None,
        verifier_model: BaseChatModel | None = None,
    ) -> None:
        self.session = session
        self.user_id = user_id
        self.embedder = embedder
        self.cache = cache or EmbeddingCache()
        self.conversations = ConversationRepository(session)
        self.chat_model = chat_model
        self.verifier_model = verifier_model

    # -- conversation CRUD -------------------------------------------------

    async def create(self, *, title: str | None) -> Conversation:
        conversation = await self.conversations.create(
            user_id=self.user_id, title=(title or DEFAULT_TITLE).strip() or DEFAULT_TITLE
        )
        await self.session.commit()
        return conversation

    async def list(
        self, *, page: int, page_size: int
    ) -> tuple[list[tuple[Conversation, int]], int]:
        page_size = max(1, min(page_size, settings.agent_max_conversations_page))
        rows, total = await self.conversations.list_for_user(
            user_id=self.user_id, offset=(page - 1) * page_size, limit=page_size
        )
        return rows, total

    async def detail(
        self, conversation_id: uuid.UUID
    ) -> tuple[Conversation, list[ConversationMessage]]:
        conversation = await self._get_or_404(conversation_id)
        messages = await self.conversations.messages(conversation.id)
        return conversation, messages

    async def delete(self, conversation_id: uuid.UUID) -> None:
        conversation = await self._get_or_404(conversation_id)
        await self.conversations.delete(conversation)
        await self.session.commit()

    async def _get_or_404(self, conversation_id: uuid.UUID) -> Conversation:
        conversation = await self.conversations.get(conversation_id, user_id=self.user_id)
        if conversation is None:
            raise NotFoundError("Conversation not found.")
        return conversation

    # -- one agent turn ----------------------------------------------------

    async def prepare_turn(self, *, conversation_id: uuid.UUID, content: str) -> PreparedTurn:
        content = content.strip()
        if len(content) < 2:
            raise ValidationError("Enter at least two characters.")
        if len(content) > settings.agent_max_message_chars:
            raise ValidationError(
                f"Messages are limited to {settings.agent_max_message_chars} characters."
            )

        conversation = await self._get_or_404(conversation_id)

        chat_model = self.chat_model or build_chat_model(streaming=True)
        if chat_model is None:
            raise ServiceUnavailableError(_NOT_CONFIGURED)
        # The reviewer answers with a short JSON object, but thinking models spend
        # completion tokens on reasoning first; too small a budget truncates the
        # content and every verdict degrades to PASS.
        verifier_model = self.verifier_model or build_chat_model(
            temperature=0.0, max_tokens=1_000, streaming=False
        )
        if verifier_model is None:
            raise ServiceUnavailableError(_NOT_CONFIGURED)

        rows = await self.conversations.messages(
            conversation.id, limit=settings.agent_max_history_messages
        )
        history: list[BaseMessage] = []
        for row in rows:
            if row.role is MessageRole.USER:
                history.append(HumanMessage(content=row.content))
            elif row.content.strip():
                history.append(AIMessage(content=row.content))
        history.append(HumanMessage(content=content))

        user_message = await self.conversations.add_message(
            conversation_id=conversation.id, role=MessageRole.USER, content=content
        )
        title = conversation.title
        if title == DEFAULT_TITLE:
            title = _derive_title(content)
            await self.conversations.rename(conversation, title)
        await self.conversations.touch(conversation)
        await self.session.commit()

        return PreparedTurn(
            conversation_id=conversation.id,
            user_message_id=user_message.id,
            title=title,
            history=history,
            chat_model=chat_model,
            verifier_model=verifier_model,
            model_name=settings.llm_model,
        )

    async def stream_turn(self, turn: PreparedTurn) -> AsyncIterator[AgentEvent]:
        started = time.perf_counter()
        answer_parts: list[str] = []
        last_answer: str | None = None
        tool_names: dict[str, str] = {}
        warning: str | None = None
        failed = False

        yield "start", {
            "conversation_id": str(turn.conversation_id),
            "user_message_id": str(turn.user_message_id),
            "title": turn.title,
            "model": turn.model_name,
        }

        async with SessionLocal() as session:
            conversations = ConversationRepository(session)
            conversation = await conversations.get(turn.conversation_id, user_id=self.user_id)
            if conversation is None:
                yield "error", {"detail": "Conversation not found.", "code": "not_found"}
                return

            context = ToolContext(
                user_id=self.user_id,
                search=SearchService(session, embedder=self.embedder, cache=self.cache),
                documents=DocumentRepository(session),
            )
            graph = build_agent_graph(
                turn.chat_model,
                turn.verifier_model,
                build_tools(context),
                context,
                max_steps=settings.agent_max_steps,
                max_verify_rounds=settings.agent_max_verify_rounds,
                verification_enabled=settings.agent_verification_enabled,
            )

            answer = ""
            citations: list[dict[str, Any]] = []
            warnings: list[str] = []
            verify_warning: str | None = None
            assistant_message_id: uuid.UUID | None = None
            # Persist in ``finally``: a disconnect arrives as GeneratorExit or
            # CancelledError, which bypass ``except Exception`` and would otherwise
            # throw the partial answer away.
            try:
                try:
                    async with asyncio.timeout(settings.agent_stream_timeout_seconds):
                        async for mode, payload in graph.astream(
                            {
                                "messages": turn.history,
                                "verdict": None,
                                "verify_attempts": 0,
                                "feedback": None,
                            },
                            stream_mode=["messages", "updates"],
                        ):
                            if mode == "messages":
                                chunk, metadata = payload
                                if (
                                    metadata.get("langgraph_node") == "agent"
                                    and isinstance(chunk, AIMessageChunk)
                                ):
                                    text = _message_text(chunk)
                                    if text:
                                        answer_parts.append(text)
                                        yield "token", {"text": text}
                                continue

                            for node, update in payload.items():
                                if node == "agent":
                                    for message in update.get("messages") or []:
                                        if not isinstance(message, AIMessage):
                                            continue
                                        if message.tool_calls:
                                            # Any preamble streamed before the tool call
                                            # is narration, not the answer; drop it so the
                                            # bubble shows only the final response.
                                            if answer_parts:
                                                answer_parts.clear()
                                                yield "answer_reset", {
                                                    "reason": "",
                                                    "attempt": 0,
                                                }
                                            for call in message.tool_calls:
                                                tool_names[call["id"]] = call["name"]
                                                yield "tool_call", {
                                                    "id": call["id"],
                                                    "name": call["name"],
                                                    "arguments": call.get("args") or {},
                                                }
                                        else:
                                            last_answer = _message_text(message).strip()
                                            if settings.agent_verification_enabled:
                                                yield "status", {"stage": "verifying"}
                                elif node == "tools":
                                    for message in update.get("messages") or []:
                                        if isinstance(message, ToolMessage):
                                            ok = not _tool_failed(message.content)
                                            yield "tool_result", {
                                                "id": message.tool_call_id,
                                                "name": message.name
                                                or tool_names.get(message.tool_call_id, "tool"),
                                                "ok": ok,
                                                "summary": _tool_summary(message.content, ok),
                                            }
                                elif node == "verify":
                                    if update.get("warning"):
                                        verify_warning = update["warning"]
                                    verdict = update.get("verdict")
                                    if verdict == "REVISE":
                                        answer_parts.clear()
                                        last_answer = None
                                        yield "answer_reset", {
                                            "reason": update.get("feedback") or "",
                                            "attempt": update.get("verify_attempts", 1),
                                        }
                                    elif verdict == "PASS":
                                        yield "status", {"stage": "verified"}
                except TimeoutError:
                    logger.warning("Assistant turn timed out for %s", self.user_id)
                    warning = _TIMEOUT
                    failed = True
                except Exception:
                    logger.warning("Assistant turn failed for %s", self.user_id, exc_info=True)
                    warning = _FAILED
                    failed = True
            finally:
                answer = (
                    last_answer if last_answer is not None else "".join(answer_parts)
                ).strip()
                citations, citation_warning = self._citations(answer, context)
                warnings = [citation_warning] if citation_warning else []
                if verify_warning:
                    warnings.insert(0, verify_warning)
                if warning:
                    warnings.insert(0, warning)
                elif not answer:
                    warnings.append(_STEP_LIMIT)
                try:
                    message = await self._persist_answer(
                        conversations,
                        conversation,
                        context,
                        answer=answer,
                        citations=citations,
                        warning=" ".join(warnings) or None,
                    )
                    assistant_message_id = message.id
                except Exception:
                    logger.warning("Could not persist assistant message", exc_info=True)
                    failed = True
                    if not warning:
                        warning = _FAILED
                        warnings.insert(0, _FAILED)

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if citations:
            yield "citations", {"citations": citations}
        if not failed:
            yield "done", {
                "message_id": str(assistant_message_id) if assistant_message_id else None,
                "elapsed_ms": elapsed_ms,
                "model": turn.model_name,
                "warning": " ".join(warnings) or None,
            }
        else:
            yield "error", {"detail": warning or _FAILED, "code": "assistant_failed"}

        logger.info(
            "Assistant turn for %s: answer=%s citations=%d tools=%d in %d ms",
            self.user_id,
            bool(answer),
            len(citations),
            len(context.tool_calls),
            elapsed_ms,
        )

    # -- helpers -----------------------------------------------------------

    def _citations(
        self, answer: str, context: ToolContext
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not answer:
            return [], None
        markers = extract_citation_markers(answer)
        valid = [marker for marker in markers if 1 <= marker <= len(context.passages)]
        invalid = [marker for marker in markers if marker not in valid]
        citations = [context.passages[marker - 1].as_citation() for marker in valid]
        if invalid:
            return citations, _INVALID_CITATIONS
        if not citations and not answer.lower().startswith(_IDK_PREFIX):
            return citations, _NO_CITATIONS
        return citations, None

    async def _persist_answer(
        self,
        conversations: ConversationRepository,
        conversation: Conversation,
        context: ToolContext,
        *,
        answer: str,
        citations: list[dict[str, Any]],
        warning: str | None,
    ) -> ConversationMessage:
        message = await conversations.add_message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            tool_calls=context.tool_calls or None,
            citations=citations or None,
            model=settings.llm_model,
            warning=warning,
        )
        await conversations.touch(conversation)
        await conversations.commit()
        return message


def _tool_summary(content: Any, ok: bool) -> str:
    if not isinstance(content, str):
        return "completed" if ok else "failed"
    if not ok:
        try:
            payload = json.loads(content)
            return str(payload.get("error", "failed"))[:200]
        except json.JSONDecodeError:
            return "failed"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content[:200]
    if isinstance(payload, dict):
        for key in ("count", "title"):
            if key in payload:
                return f"{key}={payload[key]}"
    return "completed"
