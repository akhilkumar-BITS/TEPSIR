"""The DeepSeek client must echo ``reasoning_content`` back on later turns.

DeepSeek's thinking models reject a follow-up request when an assistant message that
carried reasoning is replayed without it, which is what happens on the second call of a
tool loop. ``ChatDeepSeek`` keeps the field in ``additional_kwargs`` but drops it from
the request; ``DeepSeekChatModel`` restores it.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.model import DeepSeekChatModel


def _model() -> DeepSeekChatModel:
    return DeepSeekChatModel(
        model="deepseek-flash",
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
    )


def test_reasoning_content_is_replayed_on_assistant_turns():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="question"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "first thoughts"},
            tool_calls=[{"name": "search_policy", "args": {"query": "hotel"}, "id": "c1"}],
        ),
        ToolMessage(content='{"count": 1}', tool_call_id="c1"),
        AIMessage(content="answer", additional_kwargs={"reasoning_content": "final thoughts"}),
    ]

    payload = _model()._get_request_payload(messages)

    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert payload["messages"][2]["reasoning_content"] == "first thoughts"
    assert payload["messages"][2]["tool_calls"]
    assert payload["messages"][4]["reasoning_content"] == "final thoughts"


def test_messages_without_reasoning_are_untouched():
    payload = _model()._get_request_payload([HumanMessage(content="question")])

    assert payload["messages"] == [{"role": "user", "content": "question"}]
