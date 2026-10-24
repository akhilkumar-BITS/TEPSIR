"""Paragraph-aware, page-aware chunking.

Character windows are simpler than token windows and good enough here: policy
documents are prose, and embedding models truncate at far more characters than
``chunk_size``. Overlap preserves sentences that straddle a boundary.
"""

from dataclasses import dataclass
from itertools import count

from app.rag.extract import ExtractedPage


@dataclass(slots=True)
class TextChunk:
    index: int
    content: str
    page_number: int | None


def chunk_pages(
    pages: list[ExtractedPage], *, chunk_size: int, overlap: int
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[TextChunk] = []
    indexes = count()

    for page in pages:
        text = _normalise(page.text)
        if not text:
            continue
        for piece in _windows(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                TextChunk(index=next(indexes), content=piece, page_number=page.page_number)
            )

    return chunks


def _normalise(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    # Collapse runs of blank lines but keep paragraph breaks for readability.
    collapsed: list[str] = []
    for line in lines:
        if not line and collapsed and not collapsed[-1]:
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip()


def _windows(text: str, *, chunk_size: int, overlap: int):
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Prefer breaking after a paragraph, then a sentence, then a space.
            for boundary in ("\n\n", "\n", ". ", " "):
                candidate = text.rfind(boundary, start + chunk_size // 2, end)
                if candidate != -1:
                    end = candidate + len(boundary)
                    break
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= length:
            break
        start = max(end - overlap, start + 1)
