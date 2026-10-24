"""Embedding providers.

The interface is async because real providers (local models or hosted APIs) are
blocking or network-bound. Two implementations ship:

* ``DevHashEmbedding`` - a deterministic, dependency-free lexical embedder used
  as the default so the stack boots without downloading model weights. It is a
  hashing bag-of-words with cosine-friendly normalisation; adequate for
  smoke-testing the pipeline, not for evaluating retrieval quality.
* ``SentenceTransformerEmbedding`` - the intended embedder for the study. Enable
  with ``EMBEDDING_PROVIDER=sentence-transformers``.
"""

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

from app.core.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _validate_dimension(vector: list[float], dimension: int) -> list[float]:
    if len(vector) != dimension:
        raise ValueError(f"Embedding dimension mismatch: got {len(vector)}, expected {dimension}")
    return vector


class DevHashEmbedding:
    """Hashing-trick embedding over word unigrams and bigrams."""

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or settings.embedding_dimension
        self.name = f"dev-hash-{self.dimension}"

    def _vector(self, text: str) -> list[float]:
        tokens = _TOKEN_RE.findall(text.lower())
        terms = Counter(tokens)
        terms.update(f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False))
        if not terms:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension
        for term, count in terms.items():
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimension
            # Sub-linear term frequency keeps repeated boilerplate from dominating.
            vector[bucket] += 1.0 + math.log(count)

        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class SentenceTransformerEmbedding:
    """Local sentence-transformers model, loaded lazily on first use."""

    def __init__(self, model_name: str | None = None) -> None:
        self.name = model_name or settings.embedding_model
        self.dimension = settings.embedding_dimension
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install with: pip install '.[local-embeddings]'"
                ) from exc
            self._model = SentenceTransformer(self.name)
            model_dimension = self._model.get_sentence_embedding_dimension()
            if model_dimension != self.dimension:
                raise RuntimeError(
                    f"Model '{self.name}' produces {model_dimension}-dimensional embeddings "
                    f"but EMBEDDING_DIMENSION={self.dimension}. Align the setting and the "
                    "database column before indexing."
                )
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from asyncio import to_thread

        def _encode() -> list[list[float]]:
            model = self._load()
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [vector.tolist() for vector in vectors]

        return await to_thread(_encode)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return _validate_dimension(vectors[0], self.dimension)


def build_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedding()
    return DevHashEmbedding()
