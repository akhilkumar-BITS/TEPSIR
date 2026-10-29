"""Tools the assistant may call.

Every tool is scoped to the caller in ``ToolContext``: document lookups use the
shared corpus, but search history is always the caller's own. Tool failures are
returned as JSON the model can read and recover from — never raised — because a
bad argument should end in an explanation, not a broken stream.

Each body holds ``context.lock``. LangGraph executes the tool calls of one model
step concurrently, and those calls share a single database session and the
citation registry, neither of which tolerates interleaving.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.agent.context import ToolContext
from app.core.config import settings
from app.core.exceptions import TEPSIRError
from app.db.models import DocumentStatus
from app.schemas.search import SearchHit

_MAX_PASSAGE_CHARS = 1_500
_MAX_DOCUMENT_CHUNKS = 5
_MAX_DOCUMENT_CHUNK_CHARS = 1_200


def _as_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def build_tools(context: ToolContext) -> list[BaseTool]:
    """Create the tool set bound to one request's identity and services."""

    @tool
    async def search_policy(query: str, top_k: int = 5) -> str:
        """Search indexed travel and expense policy documents for passages relevant
        to a question. Use this for any question about policy rules, limits,
        approvals, receipts, or procedures. Returns numbered passages with their
        source document, page and similarity score."""
        query = query.strip()
        if len(query) < 2:
            return _as_json({"error": "The query must be at least two characters."})
        top_k = max(1, min(top_k, settings.search_top_k_max))
        async with context.lock:
            try:
                response = await context.search.search(
                    user_id=context.user_id, query=query, top_k=top_k
                )
            except TEPSIRError as exc:
                context.record("search_policy", {"query": query}, False, exc.message)
                return _as_json({"error": exc.message})

            passages = []
            for hit in response.results:
                passage = context.register(hit)
                passages.append(
                    {
                        "marker": passage.marker,
                        "document_id": str(passage.document_id),
                        "document_title": passage.document_title,
                        "page_number": passage.page_number,
                        "score": passage.score,
                        "content": _truncate(passage.content, _MAX_PASSAGE_CHARS),
                    }
                )
            context.record(
                "search_policy",
                {"query": query, "top_k": top_k},
                True,
                f"{len(passages)} passage(s) found",
            )
        return _as_json({"count": len(passages), "passages": passages})

    @tool
    async def list_documents() -> str:
        """List the policy documents available in the corpus, with identifiers that
        can be passed to get_document. Use this when the question is about which
        policies exist or which document covers a topic."""
        async with context.lock:
            documents, total = await context.documents.list_documents(offset=0, limit=50)
            items = [
                {
                    "document_id": str(document.id),
                    "title": document.title,
                    "status": document.status.value,
                    "chunk_count": document.chunk_count,
                }
                for document in documents
                if document.status is DocumentStatus.INDEXED
            ]
            context.record(
                "list_documents",
                {},
                True,
                f"{len(items)} indexed document(s) of {total} total",
            )
        return _as_json({"count": len(items), "documents": items})

    @tool
    async def get_document(document_id: str, max_chunks: int = 3) -> str:
        """Read an indexed policy document by its identifier: metadata plus the
        first few passages. Use this when the user asks what a particular policy
        document says, or after list_documents to inspect one."""
        try:
            parsed_id = uuid.UUID(document_id)
        except ValueError:
            return _as_json({"error": "document_id must be a UUID returned by list_documents."})

        max_chunks = max(1, min(max_chunks, _MAX_DOCUMENT_CHUNKS))
        async with context.lock:
            document = await context.documents.get(parsed_id)
            if document is None:
                context.record(
                    "get_document", {"document_id": document_id}, False, "not found"
                )
                return _as_json({"error": "No document exists with that identifier."})
            if document.status is not DocumentStatus.INDEXED:
                context.record(
                    "get_document", {"document_id": document_id}, False, "not indexed"
                )
                return _as_json(
                    {"error": f"The document is {document.status.value}, not INDEXED."}
                )

            chunks = await context.documents.list_chunks(parsed_id)
            passages = []
            for chunk in chunks[:max_chunks]:
                hit = SearchHit(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    document_title=document.title,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    content=chunk.content,
                    score=0.0,
                )
                passage = context.register(hit)
                passages.append(
                    {
                        "marker": passage.marker,
                        "page_number": passage.page_number,
                        "content": _truncate(passage.content, _MAX_DOCUMENT_CHUNK_CHARS),
                    }
                )

            context.record(
                "get_document",
                {"document_id": document_id},
                True,
                f"read {len(passages)} of {len(chunks)} passage(s)",
            )
            payload = {
                "document_id": str(document.id),
                "title": document.title,
                "description": document.description,
                "chunk_count": document.chunk_count,
                "passages": passages,
            }
        return _as_json(payload)

    @tool
    async def search_history(limit: int = 5) -> str:
        """Look up this user's recent policy searches. Use only when the user refers
        to something they asked before, for example "what did I search last week"."""
        limit = max(1, min(limit, 10))
        async with context.lock:
            rows, _ = await context.documents.list_searches(
                user_id=context.user_id, offset=0, limit=limit
            )
            items = [
                {
                    "query": row.query,
                    "result_count": row.result_count,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]
            context.record(
                "search_history", {"limit": limit}, True, f"{len(items)} search(es)"
            )
        return _as_json({"count": len(items), "searches": items})

    return [search_policy, list_documents, get_document, search_history]
