import uuid

from app.agent.context import ToolContext
from app.schemas.search import SearchHit


def _hit(chunk_id: uuid.UUID, score: float = 0.5) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        document_title="Policy",
        chunk_index=0,
        page_number=1,
        content="text",
        score=score,
    )


def _context() -> ToolContext:
    return ToolContext(user_id=uuid.uuid4(), search=None, documents=None)  # type: ignore[arg-type]


def test_markers_follow_rank_order():
    context = _context()
    first = context.register(_hit(uuid.uuid4()))
    second = context.register(_hit(uuid.uuid4()))

    assert (first.marker, second.marker) == (1, 2)


def test_repeated_chunk_keeps_its_original_marker():
    context = _context()
    chunk_id = uuid.uuid4()
    first = context.register(_hit(chunk_id, score=0.9))
    again = context.register(_hit(chunk_id, score=0.2))

    assert again.marker == first.marker
    assert len(context.passages) == 1


def test_citation_payload_serialises_identifiers():
    context = _context()
    passage = context.register(_hit(uuid.uuid4()))
    citation = passage.as_citation()

    assert citation["marker"] == 1
    assert isinstance(citation["chunk_id"], str)
    assert isinstance(citation["document_id"], str)


def test_record_keeps_tool_trace():
    context = _context()
    context.record("search_policy", {"query": "hotel"}, True, "1 passage found")
    assert context.tool_calls == [
        {
            "name": "search_policy",
            "arguments": {"query": "hotel"},
            "ok": True,
            "summary": "1 passage found",
        }
    ]
