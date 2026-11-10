"""Embedding providers.

The interface is async because real providers (local models or hosted APIs) are
blocking or network-bound. Two implementations ship:

* ``DevHashEmbedding`` - a deterministic, dependency-free lexical embedder used
  as the default so the stack boots without downloading model weights. It is a
  hashing bag-of-words with cosine-friendly normalisation; adequate for
  smoke-testing the pipeline, not for evaluating retrieval quality.
* ``SentenceTransformerEmbedding`` - the intended embedder for the study. Enable
  with ``EMBEDDING_PROVIDER=sentence-transformers``.
* ``Qwen3Embedding`` - Qwen3-Embedding models, which are instruction-aware.
  Enable with ``EMBEDDING_PROVIDER=qwen3``. The models are MRL-trained, so
  ``EMBEDDING_DIMENSION`` may be any smaller prefix of the native size (1024 for
  the 0.6B variant); the stored vectors are truncated and renormalised.
"""

import hashlib
import math
import re
from collections import Counter
from typing import Any, Protocol

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

    # Extra keyword arguments passed to SentenceTransformer(...). Subclasses that
    # need custom model code (e.g. Qwen3) override this.
    _model_kwargs: dict[str, Any] = {}
    # Prompt name used for queries only, for instruction-aware models.
    _query_prompt: str | None = None

    def __init__(self, model_name: str | None = None) -> None:
        self.name = model_name or settings.default_embedding_model
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
            self._model = SentenceTransformer(self.name, **self._model_kwargs)
            model_dimension = self._model.get_sentence_embedding_dimension()
            if model_dimension != self.dimension:
                raise RuntimeError(
                    f"Model '{self.name}' produces {model_dimension}-dimensional embeddings "
                    f"but EMBEDDING_DIMENSION={self.dimension}. Align the setting and the "
                    "database column before indexing."
                )
        return self._model

    def _encode(self, texts: list[str], prompt_name: str | None = None) -> list[list[float]]:
        model = self._load()
        kwargs: dict[str, Any] = {"normalize_embeddings": True, "show_progress_bar": False}
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        try:
            vectors = model.encode(texts, **kwargs)
        except (ValueError, KeyError):
            # Model has no matching prompt configured; fall back to a plain encode.
            kwargs.pop("prompt_name", None)
            vectors = model.encode(texts, **kwargs)
        return [vector.tolist() for vector in vectors]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from asyncio import to_thread

        return await to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        from asyncio import to_thread

        vectors = await to_thread(self._encode, [text], self._query_prompt)
        return _validate_dimension(vectors[0], self.dimension)


class Qwen3Embedding(SentenceTransformerEmbedding):
    """Qwen3-Embedding via sentence-transformers.

    Qwen3 ships custom model code, hence ``trust_remote_code=True``. The models
    are instruction-aware: queries are encoded with the ``query`` prompt while
    document chunks are encoded plainly.
    """

    _model_kwargs = {"trust_remote_code": True}
    _query_prompt = "query"

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(model_name or settings.qwen3_embedding_model)
        # Qwen3 embeddings are MRL-trained: a prefix of the vector is still a
        # usable embedding, and sentence-transformers renormalises after
        # slicing. Passing EMBEDDING_DIMENSION lets the vector column stay
        # smaller than the model's native size (1024 for the 0.6B variant).
        self._model_kwargs = {**self._model_kwargs, "truncate_dim": self.dimension}


def build_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedding()
    if settings.embedding_provider == "qwen3":
        return Qwen3Embedding()
    return DevHashEmbedding()
