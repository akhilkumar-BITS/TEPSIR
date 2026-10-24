import math

from app.rag.embeddings import DevHashEmbedding


async def test_dev_embedder_is_deterministic_and_normalised():
    embedder = DevHashEmbedding(dimension=64)
    first = await embedder.embed_query("hotel reimbursement limit")
    second = await embedder.embed_query("hotel reimbursement limit")

    assert first == second
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-6)


async def test_dev_embedder_prefers_shared_vocabulary():
    embedder = DevHashEmbedding(dimension=512)
    query = await embedder.embed_query("hotel limit")

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    related = await embedder.embed_query("maximum hotel limit per night")
    unrelated = await embedder.embed_query("airport lounge access fees")

    assert cosine(query, related) > cosine(query, unrelated)
