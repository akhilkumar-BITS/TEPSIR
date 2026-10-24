from app.rag.chunking import chunk_pages
from app.rag.extract import ExtractedPage


def test_chunks_cover_text_and_keep_page_numbers():
    text = " ".join(f"Sentence {i} about travel policy." for i in range(120))
    pages = [ExtractedPage(page_number=1, text=text)]

    chunks = chunk_pages(pages, chunk_size=300, overlap=50)

    assert len(chunks) > 1
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    # Every chunk respects the requested ceiling.
    assert all(len(chunk.content) <= 300 for chunk in chunks)


def test_blank_pages_are_skipped():
    pages = [ExtractedPage(page_number=1, text="   \n\n "), ExtractedPage(page_number=2, text="x")]
    chunks = chunk_pages(pages, chunk_size=400, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_pages([ExtractedPage(1, "text")], chunk_size=100, overlap=100)
