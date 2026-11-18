"""Step 3: Fixed-size chunker with overlap."""

from app.services.chunking.base import Chunk, validate_sizes


class FixedSizeChunker:
    """Sliding window of `chunk_size` characters with `overlap` characters shared."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 150) -> None:
        validate_sizes(chunk_size, overlap)
        self._size = chunk_size
        self._overlap = overlap

    def split(self, text: str) -> list[Chunk]:
        step = self._size - self._overlap
        chunks: list[Chunk] = []
        for start in range(0, len(text), step):
            piece = text[start : start + self._size].strip()
            if piece:
                chunks.append(Chunk(index=len(chunks), text=piece))
            if start + self._size >= len(text):
                break
        return chunks
