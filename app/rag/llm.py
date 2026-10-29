"""LLM client for answer synthesis.

The provider speaks the OpenAI-compatible chat completions protocol (DeepSeek
here), so a single ``DeepSeekClient`` covers it and the ``LLMClient`` protocol
keeps the service testable without network access. Synthesis is optional: when
no key is configured the factory returns ``None`` and callers degrade to
passages only.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Statuses worth retrying: rate limiting and transient upstream failures.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """Base class for answer-synthesis failures."""


class LLMConfigurationError(LLMError):
    """The client was built without a usable API key."""


class LLMTransportError(LLMError):
    """The request never produced an HTTP response (connect failure or timeout)."""


class LLMHTTPError(LLMError):
    """The endpoint answered with a non-2xx status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"LLM request failed with HTTP {status_code}.")
        self.status_code = status_code


class LLMResponseError(LLMError):
    """The endpoint answered with an unusable payload."""


class LLMClient(Protocol):
    model: str

    async def complete(self, *, system: str, user: str) -> str: ...

    async def aclose(self) -> None: ...


class DeepSeekClient:
    """Async client for DeepSeek's OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        backoff_seconds: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("DEEPSEEK_API_KEY is not set.")

        self.model = model or settings.llm_model
        self._max_retries = settings.llm_max_retries if max_retries is None else max_retries
        self._backoff_seconds = backoff_seconds
        timeout = (
            timeout_seconds if timeout_seconds is not None else settings.llm_timeout_seconds
        )
        # Build the URL explicitly: httpx treats a "/v1" base path as a file
        # segment and would drop it when joining a relative path.
        self._url = f"{(base_url or settings.deepseek_base_url).rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(
                max_connections=settings.llm_max_connections,
                max_keepalive_connections=settings.llm_max_keepalive_connections,
            ),
            transport=transport,
        )

    async def complete(self, *, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_output_tokens,
            "stream": False,
        }

        attempt = 0
        while True:
            try:
                response = await self._client.post(
                    self._url, headers=self._headers, json=payload
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt >= self._max_retries:
                    raise LLMTransportError("Could not reach the LLM endpoint.") from exc
                await self._backoff(attempt)
                attempt += 1
                continue
            except httpx.TimeoutException as exc:
                # A read timeout means generation may already have run; retrying
                # would double the worst-case latency.
                raise LLMTransportError("The LLM endpoint timed out.") from exc
            except httpx.HTTPError as exc:
                raise LLMTransportError("The LLM request failed.") from exc

            if response.status_code in _RETRYABLE_STATUSES and attempt < self._max_retries:
                await self._backoff(attempt)
                attempt += 1
                continue
            if response.status_code >= 400:
                raise LLMHTTPError(response.status_code)

            return self._parse(response)

    def _parse(self, response: httpx.Response) -> str:
        try:
            data: Any = response.json()
        except ValueError as exc:
            raise LLMResponseError("The LLM returned invalid JSON.") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("The LLM response did not contain a message.") from exc

        # ``reasoning_content`` is deliberately ignored: only the final answer
        # is returned to clients.
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("The LLM returned an empty answer.")
        return content.strip()

    async def _backoff(self, attempt: int) -> None:
        if self._backoff_seconds > 0:
            await asyncio.sleep(self._backoff_seconds * 2**attempt)

    async def aclose(self) -> None:
        await self._client.aclose()


def build_llm_client() -> LLMClient | None:
    """Return a configured client, or ``None`` when synthesis is unavailable."""
    if not settings.llm_enabled:
        return None
    key = (settings.deepseek_api_key or "").strip()
    if not key:
        return None
    return DeepSeekClient(api_key=key)
