"""Cross-encoder reranking: scoring contract and search-time fallbacks."""

import math
from types import SimpleNamespace

from app.core.config import settings
from app.rag.reranking import CrossEncoderReranker, build_reranker
from app.rag.vector_store import ChunkMatch
from app.services.search_service import SearchService


class _FakeCrossEncoder:
    """Maps a passage's text to a fixed logit so ordering is predictable."""

    def __init__(self, logits: dict[str, float]) -> None:
        self.logits = logits

    def predict(self, pairs) -> list[float]:
        return [self.logits[passage] for _, passage in pairs]


class _StubReranker:
    name = "stub"

    def __init__(self, scores: list[float] | Exception) -> None:
        self._scores = scores

    async def score(self, query: str, passages) -> list[float]:
        if isinstance(self._scores, Exception):
            raise self._scores
        return self._scores


def _match(content: str, score: float) -> ChunkMatch:
    chunk = SimpleNamespace(content=content)
    document = SimpleNamespace(title="Policy")
    return ChunkMatch(chunk=chunk, document=document, score=score)


def _service(reranker) -> SearchService:
    return SearchService(None, embedder=None, reranker=reranker)


async def test_cross_encoder_squashes_logits_into_unit_interval():
    reranker = CrossEncoderReranker()
    reranker._model = _FakeCrossEncoder({"relevant": 2.0, "irrelevant": -2.0})

    scores = await reranker.score("hotel limit", ["relevant", "irrelevant"])

    assert math.isclose(scores[0], 1.0 / (1.0 + math.exp(-2.0)))
    assert math.isclose(scores[1], 1.0 / (1.0 + math.exp(2.0)))
    assert all(0.0 <= score <= 1.0 for score in scores)


async def test_rerank_reorders_candidates_and_trims_to_top_k():
    matches = [_match("weak", 0.9), _match("strong", 0.4), _match("medium", 0.6)]
    reranker = _StubReranker([0.1, 0.95, 0.5])

    ranked = await _service(reranker)._rerank("query", matches, top_k=2)

    assert [match.chunk.content for match in ranked] == ["strong", "medium"]
    # Scores are min-max scaled across the candidate set: top is 1.0 and the
    # kept runner-up keeps its relative position between worst and best.
    assert ranked[0].score == 1.0
    assert math.isclose(ranked[1].score, (0.5 - 0.1) / (0.95 - 0.1))


async def test_rerank_rescales_tiny_absolute_scores_to_readable_percentages():
    matches = [_match("a", 0.9), _match("b", 0.8)]
    # Real cross-encoder sigmoid output for weak pairs: effectively 0% after
    # rounding, which is what makes the UI show "0% match".
    reranker = _StubReranker([0.00002, 0.00004])

    ranked = await _service(reranker)._rerank("query", matches, top_k=2)

    assert [match.chunk.content for match in ranked] == ["b", "a"]
    assert [match.score for match in ranked] == [1.0, 0.0]


async def test_rerank_keeps_raw_scores_when_all_candidates_tie():
    matches = [_match("a", 0.9), _match("b", 0.8)]
    reranker = _StubReranker([0.4, 0.4])

    ranked = await _service(reranker)._rerank("query", matches, top_k=2)

    assert [match.score for match in ranked] == [0.4, 0.4]


async def test_rerank_falls_back_to_vector_order_when_model_fails():
    matches = [_match("a", 0.9), _match("b", 0.8), _match("c", 0.7)]
    reranker = _StubReranker(RuntimeError("model unavailable"))

    ranked = await _service(reranker)._rerank("query", matches, top_k=2)

    assert [match.chunk.content for match in ranked] == ["a", "b"]
    assert [match.score for match in ranked] == [0.9, 0.8]


async def test_rerank_falls_back_when_score_count_mismatches():
    matches = [_match("a", 0.9), _match("b", 0.8)]
    reranker = _StubReranker([0.5])

    ranked = await _service(reranker)._rerank("query", matches, top_k=2)

    assert [match.chunk.content for match in ranked] == ["a", "b"]


def test_build_reranker_respects_setting(monkeypatch):
    monkeypatch.setattr(settings, "reranker_enabled", False)
    assert build_reranker() is None

    monkeypatch.setattr(settings, "reranker_enabled", True)
    built = build_reranker()
    assert isinstance(built, CrossEncoderReranker)
    assert built.name == settings.reranker_model
