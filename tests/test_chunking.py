import pytest

from app.core.enums import ChunkingStrategy
from app.services.chunking import get_chunker
from app.services.chunking.fixed import FixedSizeChunker
from app.services.chunking.recursive import RecursiveChunker


def test_fixed_chunker_respects_size_and_overlap() -> None:
    text = "".join(f"{i:04d}" for i in range(1000))  # 4000 chars, no whitespace
    chunks = FixedSizeChunker(chunk_size=1000, overlap=200).split(text)

    assert len(chunks) == 5
    assert all(len(chunk.text) <= 1000 for chunk in chunks)
    assert chunks[0].text[-200:] == chunks[1].text[:200]
    assert [chunk.index for chunk in chunks] == list(range(5))


def test_fixed_chunker_short_text_is_single_chunk() -> None:
    chunks = FixedSizeChunker(chunk_size=500, overlap=50).split("hello world")
    assert [chunk.text for chunk in chunks] == ["hello world"]


def test_recursive_chunker_bounds_size_and_keeps_all_content() -> None:
    paragraphs = [f"Paragraph {i}. " + "word " * 40 for i in range(30)]
    chunks = RecursiveChunker(chunk_size=300, overlap=50).split("\n\n".join(paragraphs))

    assert all(len(chunk.text) <= 300 for chunk in chunks)
    joined = "\n".join(chunk.text for chunk in chunks)
    assert all(f"Paragraph {i}." in joined for i in range(30))


def test_recursive_chunker_splits_oversized_unbroken_text() -> None:
    chunks = RecursiveChunker(chunk_size=100, overlap=10).split("x" * 450)
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert sum(len(chunk.text) for chunk in chunks) >= 450


def test_empty_text_yields_no_chunks() -> None:
    assert FixedSizeChunker().split("   ") == []
    assert RecursiveChunker().split("") == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_sizes_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=size, overlap=overlap)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=size, overlap=overlap)


def test_factory_returns_requested_strategy() -> None:
    fixed = get_chunker(ChunkingStrategy.FIXED, chunk_size=100, overlap=10)
    recursive = get_chunker(ChunkingStrategy.RECURSIVE, chunk_size=100, overlap=10)
    assert isinstance(fixed, FixedSizeChunker)
    assert isinstance(recursive, RecursiveChunker)
