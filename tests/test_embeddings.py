import math

from app.core.config import settings
from app.rag.embeddings import (
    DevHashEmbedding,
    Qwen3Embedding,
    SentenceTransformerEmbedding,
    build_embedding_provider,
)


async def test_dev_embedder_is_deterministic_and_normalised():
    embedder = DevHashEmbedding(dimension=64)
    first = await embedder.embed_query("hotel reimbursement limit")
    second = await embedder.embed_query("hotel reimbursement limit")

    assert first == second
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-6)


async def test_dev_embedder_prefers_shared_vocabulary():
    embedder = DevHashEmbedding(dimension=512)
    query = await embedder.embed_query("hotel limit")

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    related = await embedder.embed_query("maximum hotel limit per night")
    unrelated = await embedder.embed_query("airport lounge access fees")

    assert cosine(query, related) > cosine(query, unrelated)


class _RecordingModel:
    """Stands in for a SentenceTransformer and records encode options."""

    dimension = 8

    def __init__(self) -> None:
        self.encode_calls: list[dict] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: list[str], **kwargs):
        self.encode_calls.append(kwargs)
        import numpy as np

        return np.array([[1.0] + [0.0] * (self.dimension - 1) for _ in texts])


async def test_qwen3_uses_the_query_prompt_for_queries_only(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dimension", 8)
    embedder = Qwen3Embedding()
    model = _RecordingModel()
    embedder._model = model

    await embedder.embed_query("hotel limit")
    await embedder.embed_documents(["chunk text"])

    assert model.encode_calls[0]["prompt_name"] == "query"
    assert "prompt_name" not in model.encode_calls[1]


async def test_qwen3_falls_back_when_prompt_is_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dimension", 8)

    class _NoPromptModel(_RecordingModel):
        def encode(self, texts: list[str], **kwargs):
            if "prompt_name" in kwargs:
                raise ValueError("prompt not found")
            return super().encode(texts, **kwargs)

    embedder = Qwen3Embedding()
    embedder._model = _NoPromptModel()

    vector = await embedder.embed_query("hotel limit")

    assert len(vector) == 8


def test_qwen3_truncates_embeddings_to_the_configured_dimension(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dimension", 384)

    embedder = Qwen3Embedding()

    assert embedder._model_kwargs["truncate_dim"] == 384
    assert embedder._model_kwargs["trust_remote_code"] is True


def test_build_embedding_provider_selects_qwen3(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "qwen3")
    assert isinstance(build_embedding_provider(), Qwen3Embedding)

    monkeypatch.setattr(settings, "embedding_provider", "sentence-transformers")
    assert isinstance(build_embedding_provider(), SentenceTransformerEmbedding)
    assert not isinstance(build_embedding_provider(), Qwen3Embedding)
