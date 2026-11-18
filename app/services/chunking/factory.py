from app.core.enums import ChunkingStrategy
from app.services.chunking.base import Chunker
from app.services.chunking.fixed import FixedSizeChunker
from app.services.chunking.recursive import RecursiveChunker


def get_chunker(strategy: ChunkingStrategy, *, chunk_size: int, overlap: int) -> Chunker:
    """Return the chunker implementing the requested strategy."""
    if strategy is ChunkingStrategy.FIXED:
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy is ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")
