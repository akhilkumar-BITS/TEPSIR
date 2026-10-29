import uuid

from app.rag.embeddings import DevHashEmbedding
from app.schemas.search import SearchHit, SearchResponse
from app.services.answer_service import AnswerService


def make_hit(index: int, content: str = "Hotel cap is 200 USD per night.") -> SearchHit:
    return SearchHit(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title=f"Policy {index}",
        chunk_index=index,
        page_number=index,
        content=content,
        score=round(1.0 - index / 100, 4),
    )


class StubSearchService:
    def __init__(self, results: list[SearchHit]) -> None:
        self.results = results
        self.calls = 0

    async def search(self, *, user_id, query, top_k) -> SearchResponse:
        self.calls += 1
        return SearchResponse(query=query, results=self.results, elapsed_ms=1)


class FakeLLM:
    model = "fake-model"

    def __init__(self, *, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls = 0

    async def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.text

    async def aclose(self) -> None:
        return None


def build_service(results, llm) -> tuple[AnswerService, StubSearchService]:
    search = StubSearchService(results)
    service = AnswerService(
        None,
        embedder=DevHashEmbedding(dimension=8),
        llm=llm,
        search_service=search,
    )
    return service, search


async def test_without_llm_returns_passages_and_a_warning():
    hits = [make_hit(1), make_hit(2)]
    service, _ = build_service(hits, llm=None)

    response = await service.answer(user_id=uuid.uuid4(), query="hotel cap", top_k=2)

    assert response.answer is None
    assert response.citations == []
    assert response.results == hits
    assert response.model is None
    assert response.warning is not None
    assert "not configured" in response.warning


async def test_llm_failure_degrades_to_passages():
    hits = [make_hit(1)]
    llm = FakeLLM(error=RuntimeError("provider down"))
    service, _ = build_service(hits, llm)

    response = await service.answer(user_id=uuid.uuid4(), query="hotel cap", top_k=1)

    assert response.answer is None
    assert response.results == hits
    assert response.model == "fake-model"
    assert response.warning is not None
    assert "unavailable" in response.warning
    assert llm.calls == 1


async def test_citation_marker_maps_to_the_retrieved_hit():
    hits = [make_hit(1), make_hit(2)]
    service, _ = build_service(hits, FakeLLM(text="The cap is 200 USD [1]."))

    response = await service.answer(user_id=uuid.uuid4(), query="hotel cap", top_k=2)

    assert response.answer == "The cap is 200 USD [1]."
    assert len(response.citations) == 1
    citation = response.citations[0]
    assert citation.marker == 1
    assert citation.chunk_id == hits[0].chunk_id
    assert citation.document_title == hits[0].document_title
    assert response.warning is None


async def test_second_marker_maps_to_the_second_hit():
    hits = [make_hit(1), make_hit(2)]
    service, _ = build_service(hits, FakeLLM(text="Receipts are required [2]."))

    response = await service.answer(user_id=uuid.uuid4(), query="receipts", top_k=2)

    assert [c.chunk_id for c in response.citations] == [hits[1].chunk_id]


async def test_unknown_marker_is_dropped_and_reported():
    hits = [make_hit(1)]
    service, _ = build_service(hits, FakeLLM(text="Cap is 200 [1] and also [9]."))

    response = await service.answer(user_id=uuid.uuid4(), query="hotel cap", top_k=1)

    assert [c.marker for c in response.citations] == [1]
    assert response.warning is not None
    assert "not provided" in response.warning


async def test_no_results_skips_the_llm_call():
    llm = FakeLLM(text="should not be used")
    service, search = build_service([], llm)

    response = await service.answer(user_id=uuid.uuid4(), query="unrelated", top_k=3)

    assert search.calls == 1
    assert llm.calls == 0
    assert response.results == []
    assert response.answer is None
    assert response.warning is not None
    assert "No matching" in response.warning


async def test_dont_know_answer_does_not_warn_about_citations():
    hits = [make_hit(1)]
    text = "I don't know based on the available policy passages."
    service, _ = build_service(hits, FakeLLM(text=text))

    response = await service.answer(user_id=uuid.uuid4(), query="hotel cap", top_k=1)

    assert response.answer == text
    assert response.citations == []
    assert response.warning is None
