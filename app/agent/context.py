"""Per-turn state shared by the agent's tools.

Tools are plain functions to the graph, so the request's identity, services and
the passages seen so far travel in a ``ToolContext`` that the tool closures
capture. Citation markers are assigned here once and reused when the same chunk
comes back from a later search, so ``[n]`` stays stable for the model.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.repositories.document_repository import DocumentRepository
from app.schemas.search import SearchHit
from app.services.search_service import SearchService


@dataclass(slots=True)
class RegisteredPassage:
    marker: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page_number: int | None
    score: float
    content: str

    def as_citation(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "document_title": self.document_title,
            "page_number": self.page_number,
            "score": self.score,
        }


@dataclass
class ToolContext:
    user_id: uuid.UUID
    search: SearchService
    documents: DocumentRepository
    passages: list[RegisteredPassage] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    _markers: dict[uuid.UUID, int] = field(default_factory=dict)
    # The model may ask for several tools in one step and LangGraph runs them
    # concurrently. They share one database session, which cannot flush twice at
    # once, and they assign citation markers in execution order, so they take
    # turns instead.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def register(self, hit: SearchHit) -> RegisteredPassage:
        """Return the passage for ``hit``, numbering it on first sight."""
        existing = self._markers.get(hit.chunk_id)
        if existing is not None:
            return self.passages[existing - 1]
        passage = RegisteredPassage(
            marker=len(self.passages) + 1,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            document_title=hit.document_title,
            page_number=hit.page_number,
            score=hit.score,
            content=hit.content,
        )
        self.passages.append(passage)
        self._markers[hit.chunk_id] = passage.marker
        return passage

    def record(self, name: str, arguments: dict[str, Any], ok: bool, summary: str) -> None:
        self.tool_calls.append(
            {"name": name, "arguments": arguments, "ok": ok, "summary": summary}
        )
