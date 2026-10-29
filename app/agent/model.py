"""Chat-model factory for the agent.

DeepSeek's API is OpenAI-compatible but its thinking models add a
``reasoning_content`` field that must be echoed back on every assistant turn.
``ChatDeepSeek`` parses the field and keeps it in ``additional_kwargs``, yet the
request payload drops it, so the second model call of a tool loop fails with
HTTP 400. The subclass below puts it back. The client is built per request
because the graph and its tools are per request too; the pooled HTTP connections
inside the client keep that cheap.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from app.core.config import settings


class DeepSeekChatModel(ChatDeepSeek):
    """``ChatDeepSeek`` that replays ``reasoning_content`` to the API."""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for source, target in zip(messages, payload.get("messages", []), strict=False):
            if not isinstance(source, AIMessage):
                continue
            reasoning = source.additional_kwargs.get("reasoning_content")
            if reasoning and target.get("role") == "assistant":
                target["reasoning_content"] = reasoning
        return payload


def build_chat_model(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = True,
) -> DeepSeekChatModel | None:
    """Return a client, or None when the agent is disabled or has no key."""
    if not settings.agent_enabled or not settings.llm_enabled:
        return None
    api_key = (settings.deepseek_api_key or "").strip()
    if not api_key:
        return None
    return DeepSeekChatModel(
        model=settings.llm_model,
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_output_tokens if max_tokens is None else max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        streaming=streaming,
        stream_usage=False,
    )
