"""Answer synthesis over retrieved policy passages.

Retrieval stays the source of truth: this service asks ``SearchService`` for
ranked passages, then optionally turns them into a cited answer. Every failure
in the optional LLM step degrades to "passages only" with a warning instead of
failing the request, so search keeps working without an API key.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import EmbeddingCache
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import LLMClient
from app.rag.prompting import PassageSource, build_answer_prompt, select_citations
from app.schemas.search import AnswerResponse, Citation, SearchHit
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

_NOT_CONFIGURED = "Answer generation is not configured. Showing retrieved passages instead."
_UNAVAILABLE = "Answer generation is unavailable right now. Showing retrieved passages instead."
_CONTEXT_TOO_SMALL = (
    "The retrieved passages did not fit within the answer context window."
)
_NO_RESULTS = "No matching policy passages were found."
_INVALID_CITATIONS = (
    "The answer referenced passages that were not provided; those citations were ignored."
)
_NO_CITATIONS = "The generated answer did not cite any passages; verify it against the results."


@dataclass(slots=True)
class SynthesisResult:
    answer: str | None
    citations: list[Citation]
    model: str | None
    warning: str | None


class AnswerService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        embedder: EmbeddingProvider,
        cache: EmbeddingCache | None = None,
        llm: LLMClient | None = None,
        search_service: SearchService | None = None,
    ) -> None:
        self.search = search_service or SearchService(session, embedder=embedder, cache=cache)
        self.llm = llm

    async def answer(
        self, *, user_id: uuid.UUID, query: str, top_k: int
    ) -> AnswerResponse:
        started = time.perf_counter()
        retrieval = await self.search.search(user_id=user_id, query=query, top_k=top_k)

        if not retrieval.results:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info("Answer for %s: no passages in %d ms", user_id, elapsed_ms)
            return AnswerResponse(
                query=retrieval.query,
                answer=None,
                citations=[],
                results=[],
                model=None,
                elapsed_ms=elapsed_ms,
                warning=_NO_RESULTS,
            )

        synthesis = await self.synthesize(query=retrieval.query, hits=retrieval.results)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Answer for %s: answered=%s citations=%d in %d ms",
            user_id,
            synthesis.answer is not None,
            len(synthesis.citations),
            elapsed_ms,
        )
        return AnswerResponse(
            query=retrieval.query,
            answer=synthesis.answer,
            citations=synthesis.citations,
            results=retrieval.results,
            model=synthesis.model,
            elapsed_ms=elapsed_ms,
            warning=synthesis.warning,
        )

    async def synthesize(
        self, *, query: str, hits: Sequence[SearchHit]
    ) -> SynthesisResult:
        if self.llm is None:
            return SynthesisResult(answer=None, citations=[], model=None, warning=_NOT_CONFIGURED)

        prompt = build_answer_prompt(
            query,
            [
                PassageSource(
                    document_title=hit.document_title,
                    page_number=hit.page_number,
                    content=hit.content,
                )
                for hit in hits
            ],
        )
        if not prompt.passages:
            return SynthesisResult(
                answer=None, citations=[], model=self.llm.model, warning=_CONTEXT_TOO_SMALL
            )

        try:
            raw = await self.llm.complete(system=prompt.system, user=prompt.user)
        except Exception:
            # Synthesis is optional: a provider outage must not break search.
            logger.warning("Answer synthesis failed", exc_info=True)
            return SynthesisResult(
                answer=None, citations=[], model=self.llm.model, warning=_UNAVAILABLE
            )

        answer = raw.strip()
        if not answer:
            return SynthesisResult(
                answer=None, citations=[], model=self.llm.model, warning=_UNAVAILABLE
            )

        selected, invalid = select_citations(answer, prompt.passages)
        citations = [
            Citation(
                marker=passage.marker,
                chunk_id=hits[passage.marker - 1].chunk_id,
                document_id=hits[passage.marker - 1].document_id,
                document_title=passage.document_title,
                page_number=passage.page_number,
                score=hits[passage.marker - 1].score,
            )
            for passage in selected
        ]

        warnings: list[str] = []
        if invalid:
            warnings.append(_INVALID_CITATIONS)
        if prompt.omitted_count:
            warnings.append(
                f"{prompt.omitted_count} lower-ranked passage(s) were omitted to fit the "
                "model context window."
            )
        if not citations and not answer.lower().startswith("i don't know"):
            warnings.append(_NO_CITATIONS)

        return SynthesisResult(
            answer=answer,
            citations=citations,
            model=self.llm.model,
            warning=" ".join(warnings) or None,
        )
