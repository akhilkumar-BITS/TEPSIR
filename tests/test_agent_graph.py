import json
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from app.agent.context import ToolContext
from app.agent.graph import build_agent_graph


class ScriptedChatModel(BaseChatModel):
    """Returns canned messages in order and records every prompt it received."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.calls.append(list(messages))
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


@tool
async def search_policy(query: str) -> str:
    """Search policy passages."""
    return json.dumps({"count": 1, "passages": [{"marker": 1, "content": "Hotel limit 150."}]})


def _revise() -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {"verdict": "REVISE", "reason": "Receipt rule missing.", "next_query": "receipts"}
        )
    )


def _pass() -> AIMessage:
    return AIMessage(content='{"verdict": "PASS", "reason": "Supported.", "next_query": null}')


def _context() -> ToolContext:
    return ToolContext(user_id=uuid.uuid4(), search=None, documents=None)  # type: ignore[arg-type]


async def _run(agent_messages: list[AIMessage], verifier_messages: list[AIMessage], **kwargs: Any):
    agent = ScriptedChatModel(responses=agent_messages)
    verifier = ScriptedChatModel(responses=verifier_messages)
    graph = build_agent_graph(
        agent,
        verifier,
        [search_policy],
        _context(),
        max_steps=kwargs.get("max_steps", 4),
        max_verify_rounds=kwargs.get("max_verify_rounds", 1),
        verification_enabled=kwargs.get("verification_enabled", True),
    )
    state = await graph.ainvoke(
        {
            "messages": [("user", "What is the hotel limit and do I need receipts?")],
            "verdict": None,
            "verify_attempts": 0,
            "feedback": None,
        }
    )
    return agent, verifier, state


async def test_revise_verdict_sends_the_agent_back_with_feedback():
    agent, verifier, state = await _run(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_policy", "args": {"query": "hotel"}, "id": "c1"}],
            ),
            AIMessage(content="Draft answer [1]"),
            AIMessage(content="Final answer [1]"),
        ],
        [_revise(), _pass()],
    )

    assert state["messages"][-1].content == "Final answer [1]"
    assert len(verifier.calls) == 2
    follow_up = agent.calls[-1]
    assert any(
        isinstance(message, SystemMessage) and "rejected by the reviewer" in message.content
        for message in follow_up
    )


async def test_pass_verdict_stops_after_one_answer():
    agent, verifier, state = await _run(
        [AIMessage(content="Answer [1]")],
        [_pass()],
    )

    assert state["messages"][-1].content == "Answer [1]"
    assert len(verifier.calls) == 1
    assert len(agent.calls) == 1


async def test_revision_limit_accepts_the_answer_after_one_retry():
    _, verifier, state = await _run(
        [AIMessage(content="First [1]"), AIMessage(content="Second [1]")],
        [_revise(), _revise(), _revise()],
        max_verify_rounds=1,
    )

    assert state["verdict"] == "PASS"
    assert state["messages"][-1].content == "Second [1]"
    assert len(verifier.calls) == 2
    assert "reviewer still had concerns" in (state.get("warning") or "").lower()


async def test_verification_can_be_disabled():
    _, verifier, state = await _run(
        [AIMessage(content="Answer [1]")],
        [_revise()],
        verification_enabled=False,
    )

    assert verifier.calls == []
    assert state["messages"][-1].content == "Answer [1]"


async def test_tool_budget_ends_the_graph():
    def tool_call(index: int) -> AIMessage:
        return AIMessage(
            content="",
            id=f"ai-{index}",
            tool_calls=[{"name": "search_policy", "args": {"query": "x"}, "id": f"c{index}"}],
        )

    _, _, state = await _run(
        [tool_call(1), tool_call(2), tool_call(3)], [_pass()], max_steps=2
    )

    assert state["messages"][-1].tool_calls
