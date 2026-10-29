from app.rag.prompting import (
    PassageSource,
    PromptPassage,
    build_answer_prompt,
    extract_citation_markers,
    select_citations,
)


def source(title: str, content: str, page: int | None = 1) -> PassageSource:
    return PassageSource(document_title=title, page_number=page, content=content)


def test_markers_are_one_based_and_follow_rank_order():
    prompt = build_answer_prompt(
        "hotel limit",
        [source("Travel policy", "Hotel cap is 200."), source("Expenses", "Receipts required.")],
    )

    assert [p.marker for p in prompt.passages] == [1, 2]
    assert prompt.user.startswith("Question: hotel limit")
    assert prompt.user.index("[1] Travel policy") < prompt.user.index("[2] Expenses")
    assert prompt.omitted_count == 0
    assert prompt.truncated is False


def test_header_includes_page_only_when_known():
    prompt = build_answer_prompt(
        "receipts",
        [source("No page doc", "Text.", page=None), source("Paged doc", "Text.", page=4)],
    )

    headers = [line for line in prompt.user.splitlines() if line.startswith("[")]
    assert headers[0] == "[1] No page doc"
    assert headers[1] == "[2] Paged doc (page 4)"


def test_long_passage_is_truncated_with_ellipsis():
    long_text = "policy " * 200
    prompt = build_answer_prompt(
        "limit", [source("Long doc", long_text)], max_context_chars=10_000, max_passage_chars=80
    )

    passage = prompt.passages[0]
    assert len(passage.content) <= 82  # limit plus the ellipsis
    assert passage.content.endswith("…")
    assert prompt.truncated is True
    assert prompt.omitted_count == 0


def test_context_budget_omits_lower_ranked_passages():
    passages = [source(f"Doc {i}", "x" * 120) for i in range(1, 6)]
    prompt = build_answer_prompt(
        "query", passages, max_context_chars=400, max_passage_chars=1_000
    )

    assert len(prompt.passages) == 2
    assert prompt.omitted_count == 3
    assert "[3]" not in prompt.user


def test_tiny_budget_includes_nothing():
    prompt = build_answer_prompt(
        "query", [source("Doc", "x" * 200)], max_context_chars=20, max_passage_chars=200
    )

    assert prompt.passages == []
    assert prompt.omitted_count == 1
    assert "Question: query" in prompt.user


def test_extract_markers_handles_groups_and_dedupes():
    answer = "Cap is 200 [2], receipts [1] and [2] again; see [3, 4] and [4;5]."

    assert extract_citation_markers(answer) == [2, 1, 3, 4, 5]


def test_select_citations_drops_unknown_and_ids_valid_passages():
    passages = [
        PromptPassage(marker=1, document_title="A", page_number=None, content="a"),
        PromptPassage(marker=2, document_title="B", page_number=3, content="b"),
    ]

    valid, invalid = select_citations("See [2] and [9].", passages)

    assert [p.marker for p in valid] == [2]
    assert invalid == [9]


def test_select_citations_keeps_first_mention_order():
    passages = [
        PromptPassage(marker=1, document_title="A", page_number=None, content="a"),
        PromptPassage(marker=2, document_title="B", page_number=None, content="b"),
    ]

    valid, invalid = select_citations("[2][1][2]", passages)

    assert [p.marker for p in valid] == [2, 1]
    assert invalid == []
