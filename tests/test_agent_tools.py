import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

from app.agent.context import ToolContext
from app.agent.tools import build_tools
from app.db.models import DocumentStatus
from app.schemas.search import SearchHit, SearchResponse

USER_ID = uuid.uuid4()
DOC_ID = uuid.uuid4()
CHUNK_ID = uuid.uuid4()


def _hit(**overrides: object) -> SearchHit:
    data = {
        "chunk_id": CHUNK_ID,
        "document_id": DOC_ID,
        "document_title": "Travel policy",
        "chunk_index": 0,
        "page_number": 3,
        "content": "Hotel limit is 150 USD per night.",
        "score": 0.8,
    }
    data.update(overrides)
    return SearchHit(**data)


class StubSearch:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[dict[str, object]] = []

    async def search(self, *, user_id: uuid.UUID, query: str, top_k: int) -> SearchResponse:
        self.calls.append({"user_id": user_id, "query": query, "top_k": top_k})
        return SearchResponse(query=query, results=self.hits, elapsed_ms=1)


class StubDocuments:
    def __init__(
        self,
        *,
        documents: list[SimpleNamespace] | None = None,
        chunks: list[SimpleNamespace] | None = None,
        searches: list[SimpleNamespace] | None = None,
    ) -> None:
        self._documents = documents or []
        self._chunks = chunks or []
        self._searches = searches or []

    async def list_documents(self, *, offset: int, limit: int):
        return self._documents, len(self._documents)

    async def get(self, document_id: uuid.UUID):
        for document in self._documents:
            if document.id == document_id:
                return document
        return None

    async def list_chunks(self, document_id: uuid.UUID):
        return self._chunks

    async def list_searches(self, *, user_id: uuid.UUID | None, offset: int, limit: int):
        return self._searches, len(self._searches)


def _context(search=None, documents=None) -> ToolContext:
    return ToolContext(
        user_id=USER_ID,
        search=search or StubSearch([_hit()]),
        documents=documents or StubDocuments(),
    )


def _tool(tools, name):
    return next(tool for tool in tools if tool.name == name)


async def test_search_policy_returns_numbered_passages_and_records_call():
    context = _context()
    tool = _tool(build_tools(context), "search_policy")
    result = json.loads(await tool.ainvoke({"query": "hotel limit"}))

    assert result["count"] == 1
    assert result["passages"][0]["marker"] == 1
    assert result["passages"][0]["document_title"] == "Travel policy"
    assert result["passages"][0]["page_number"] == 3
    assert context.tool_calls[0]["name"] == "search_policy"
    assert context.tool_calls[0]["ok"] is True


async def test_search_policy_reuses_marker_for_same_chunk():
    context = _context()
    tools = build_tools(context)
    await _tool(tools, "search_policy").ainvoke({"query": "hotel"})
    second = json.loads(await _tool(tools, "search_policy").ainvoke({"query": "hotel again"}))

    assert second["passages"][0]["marker"] == 1
    assert len(context.passages) == 1


async def test_search_policy_clamps_top_k():
    search = StubSearch([_hit()])
    tools = build_tools(_context(search=search))
    await _tool(tools, "search_policy").ainvoke({"query": "hotel", "top_k": 500})
    assert search.calls[0]["top_k"] <= 20


async def test_search_policy_rejects_short_query():
    tool = _tool(build_tools(_context()), "search_policy")
    result = json.loads(await tool.ainvoke({"query": "a"}))
    assert "error" in result


async def test_list_documents_hides_unindexed_documents():
    documents = [
        SimpleNamespace(
            id=DOC_ID, title="Travel policy", status=DocumentStatus.INDEXED, chunk_count=4
        ),
        SimpleNamespace(
            id=uuid.uuid4(), title="Draft", status=DocumentStatus.FAILED, chunk_count=0
        ),
    ]
    context = _context(documents=StubDocuments(documents=documents))
    result = json.loads(await _tool(build_tools(context), "list_documents").ainvoke({}))

    assert result["count"] == 1
    assert result["documents"][0]["title"] == "Travel policy"


async def test_get_document_registers_passages_for_citation():
    document = SimpleNamespace(
        id=DOC_ID,
        title="Travel policy",
        description=None,
        status=DocumentStatus.INDEXED,
        chunk_count=1,
    )
    chunk = SimpleNamespace(id=CHUNK_ID, chunk_index=0, page_number=2, content="Approval needed.")
    context = _context(documents=StubDocuments(documents=[document], chunks=[chunk]))
    result = json.loads(
        await _tool(build_tools(context), "get_document").ainvoke({"document_id": str(DOC_ID)})
    )

    assert result["passages"][0]["marker"] == 1
    assert context.passages[0].chunk_id == CHUNK_ID


async def test_get_document_rejects_bad_identifier():
    result = json.loads(
        await _tool(build_tools(_context()), "get_document").ainvoke({"document_id": "nope"})
    )
    assert "error" in result


async def test_search_history_uses_the_calling_user():
    rows = [SimpleNamespace(query="hotel", result_count=3, created_at=datetime.now(UTC))]
    context = _context(documents=StubDocuments(searches=rows))
    result = json.loads(await _tool(build_tools(context), "search_history").ainvoke({"limit": 5}))

    assert result["searches"][0]["query"] == "hotel"
