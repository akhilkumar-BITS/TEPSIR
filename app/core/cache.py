"""Optional Redis cache for query embeddings.

Cache misses and Redis outages are both non-fatal: the caller simply embeds the
query as usual. That keeps Redis a performance dependency, not a correctness one.
"""

import hashlib
import json
import logging

from redis.asyncio import Redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, url: str | None = None, ttl_seconds: int | None = None) -> None:
        self._url = url if url is not None else settings.redis_url
        self._ttl = ttl_seconds or settings.embedding_cache_ttl_seconds
        self._redis: Redis | None = None

    @property
    def enabled(self) -> bool:
        return settings.search_cache_enabled and bool(self._url)

    def _client(self) -> Redis:
        if self._redis is None:
            assert self._url is not None
            self._redis = Redis.from_url(self._url, decode_responses=True)
        return self._redis

    @staticmethod
    def _key(text: str, model: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"tepsir:embedding:{model}:{digest}"

    async def get(self, text: str, model: str) -> list[float] | None:
        if not self.enabled:
            return None
        try:
            raw = await self._client().get(self._key(text, model))
        except Exception as exc:  # noqa: BLE001 - cache must never break search
            logger.debug("Embedding cache read failed: %s", exc)
            return None
        if raw is None:
            return None
        try:
            return [float(value) for value in json.loads(raw)]
        except (ValueError, TypeError):
            return None

    async def set(self, text: str, model: str, vector: list[float]) -> None:
        if not self.enabled:
            return
        try:
            await self._client().set(
                self._key(text, model), json.dumps(vector), ex=self._ttl
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Embedding cache write failed: %s", exc)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
