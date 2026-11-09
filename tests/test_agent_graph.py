import json
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from app.agent.context import RegisteredPassage, ToolContext
from app.agent.graph import _format_passages, build_agent_graph


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


def _revise_in_reasoning() -> AIMessage:
    """Thinking model that spent its whole content budget on reasoning."""
    return AIMessage(
        content="",
        additional_kwargs={
            "reasoning_content": json.dumps(
                {"verdict": "REVISE", "reason": "Receipt rule missing.", "next_query": "receipts"}
            )
        },
    )


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
    # The retry must be told not to bail out with "I don't know" while relevant
    # passages exist; that was the failure mode observed in practice.
    assert any(
        isinstance(message, SystemMessage) and "Do not reply" in message.content
        for message in follow_up
    )


def _crowded_context(count: int) -> ToolContext:
    context = _context()
    for marker in range(1, count + 1):
        context.passages.append(
            RegisteredPassage(
                marker=marker,
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                document_title="Policy",
                page_number=None,
                score=0.5,
                content=f"passage {marker} " + "x" * 900,
            )
        )
    return context


def test_verifier_context_always_includes_cited_passages():
    context = _crowded_context(12)

    # Without focus the character budget cuts the tail, so [11] never reaches
    # the reviewer and a correct citation looks hallucinated.
    assert "[11]" not in _format_passages(context)
    assert "[11]" in _format_passages(context, {11})
    assert "omitted for length" in _format_passages(context, {11})


async def test_verdict_is_recovered_from_reasoning_content():
    agent, verifier, state = await _run(
        [AIMessage(content="Draft [1]"), AIMessage(content="Final [1]")],
        [_revise_in_reasoning(), _pass()],
    )

    assert state["messages"][-1].content == "Final [1]"
    assert len(verifier.calls) == 2
    assert len(agent.calls) == 2


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


async def test_two_revisions_are_allowed_before_the_limit():
    _, verifier, state = await _run(
        [
            AIMessage(content="First [1]"),
            AIMessage(content="Second [1]"),
            AIMessage(content="Third [1]"),
        ],
        [_revise(), _revise(), _pass()],
        max_verify_rounds=2,
    )

    assert state["verdict"] == "PASS"
    assert state["messages"][-1].content == "Third [1]"
    assert len(verifier.calls) == 3
    assert state.get("warning") is None


async def test_verification_can_be_disabled():
    _, verifier, state = await _run(
        [AIMessage(content="Answer [1]")],
        [_revise()],
        verification_enabled=False,
    )

    assert verifier.calls == []
    assert state["messages"][-1].content == "Answer [1]"


async def test_revision_pass_gets_a_fresh_tool_budget():
    def tool_call(index: int) -> AIMessage:
        return AIMessage(
            content="",
            id=f"ai-{index}",
            tool_calls=[{"name": "search_policy", "args": {"query": "x"}, "id": f"c{index}"}],
        )

    # First pass uses its only allowed search, the reviewer rejects the draft,
    # and the retry must still be allowed to search before answering. With a
    # shared budget the retry's search would have ended the graph instead.
    agent, _, state = await _run(
        [
            tool_call(1),
            AIMessage(content="Draft [1]"),
            tool_call(2),
            AIMessage(content="Final [1]"),
        ],
        [_revise(), _pass()],
        max_steps=2,
    )

    assert state["messages"][-1].content == "Final [1]"
    assert state["verdict"] == "PASS"
    assert len(agent.calls) == 4


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
