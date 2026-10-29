import json

import httpx
import pytest

from app.core.config import settings
from app.rag.llm import (
    DeepSeekClient,
    LLMConfigurationError,
    LLMHTTPError,
    LLMResponseError,
    LLMTransportError,
    build_llm_client,
)


def make_client(handler, **kwargs) -> DeepSeekClient:
    return DeepSeekClient(
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-flash",
        backoff_seconds=0,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def ok_response(content: str, reasoning: str | None = "internal reasoning") -> httpx.Response:
    message: dict[str, str | None] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return httpx.Response(200, json={"choices": [{"index": 0, "message": message}]})


async def test_complete_returns_content_and_ignores_reasoning_content():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        seen["payload"] = json.loads(request.content)
        return ok_response("The nightly cap is 200 USD [1].")

    client = make_client(handler)
    try:
        answer = await client.complete(system="system text", user="user text")
    finally:
        await client.aclose()

    assert answer == "The nightly cap is 200 USD [1]."
    assert "reasoning" not in answer
    assert seen["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert seen["authorization"] == "Bearer test-key"
    payload = seen["payload"]
    assert payload["model"] == "deepseek-flash"
    assert payload["messages"][0] == {"role": "system", "content": "system text"}
    assert payload["messages"][1] == {"role": "user", "content": "user text"}
    assert payload["stream"] is False


async def test_retryable_status_is_retried_once():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="unavailable")
        return ok_response("ok")

    client = make_client(handler)
    try:
        assert await client.complete(system="s", user="u") == "ok"
    finally:
        await client.aclose()

    assert calls == 2


async def test_client_error_is_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad request")

    client = make_client(handler)
    try:
        with pytest.raises(LLMHTTPError) as excinfo:
            await client.complete(system="s", user="u")
    finally:
        await client.aclose()

    assert excinfo.value.status_code == 400
    assert calls == 1


async def test_persistent_server_error_exhausts_retries():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text="boom")

    client = make_client(handler)
    try:
        with pytest.raises(LLMHTTPError):
            await client.complete(system="s", user="u")
    finally:
        await client.aclose()

    assert calls == 2


async def test_connect_error_is_retried_then_surfaces_as_transport_error():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(handler)
    try:
        with pytest.raises(LLMTransportError):
            await client.complete(system="s", user="u")
    finally:
        await client.aclose()

    assert calls == 2


async def test_read_timeout_is_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("too slow", request=request)

    client = make_client(handler)
    try:
        with pytest.raises(LLMTransportError):
            await client.complete(system="s", user="u")
    finally:
        await client.aclose()

    assert calls == 1


async def test_invalid_json_is_a_response_error():
    client = make_client(lambda request: httpx.Response(200, text="not json"))
    try:
        with pytest.raises(LLMResponseError):
            await client.complete(system="s", user="u")
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "   "}}]},
        {"unexpected": True},
    ],
)
async def test_unusable_payloads_are_response_errors(payload):
    client = make_client(lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(LLMResponseError):
            await client.complete(system="s", user="u")
    finally:
        await client.aclose()


def test_empty_api_key_is_a_configuration_error():
    with pytest.raises(LLMConfigurationError):
        DeepSeekClient(api_key="   ")


def test_build_llm_client_returns_none_without_key(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    assert build_llm_client() is None


def test_build_llm_client_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_enabled", False)
    assert build_llm_client() is None


async def test_build_llm_client_builds_with_key(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_enabled", True)

    client = build_llm_client()
    assert isinstance(client, DeepSeekClient)
    assert client.model == settings.llm_model
    await client.aclose()
