"""Cross-encoder reranking of vector-search candidates.

Embedding search is fast but coarse: it compares independently computed vectors.
A cross-encoder reads the query and a passage together and scores their
relevance, which is more accurate but far too slow to run over the whole corpus.
So retrieval fetches a wider candidate set and the reranker reorders it.

The model is loaded lazily on first use, the same way the embedding provider is,
and reranking failures degrade to the original vector order instead of failing
the request.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from app.core.config import settings


class Reranker(Protocol):
    name: str

    async def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Sentence-transformers cross-encoder, loaded lazily on first use."""

    def __init__(self, model_name: str | None = None) -> None:
        self.name = model_name or settings.reranker_model
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install with: pip install '.[local-embeddings]'"
                ) from exc
            self._model = CrossEncoder(self.name)
        return self._model

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        from asyncio import to_thread

        def _predict() -> list[float]:
            model = self._load()
            logits = model.predict([(query, passage) for passage in passages])
            # ms-marco cross-encoders emit unbounded logits; squash to [0, 1] so
            # the API's relevance score keeps the same meaning as cosine similarity.
            return [1.0 / (1.0 + math.exp(-float(logit))) for logit in logits]

        return await to_thread(_predict)


def build_reranker() -> Reranker | None:
    """Return the configured reranker, or None when reranking is disabled."""
    if not settings.reranker_enabled:
        return None
    return CrossEncoderReranker()
