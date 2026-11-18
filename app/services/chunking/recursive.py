from collections.abc import Sequence
from typing import Final

from app.services.chunking.base import Chunk, validate_sizes

DEFAULT_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", ". ", " ")


class RecursiveChunker:
    """Structure-aware chunking.

    Splits on the coarsest separator first (paragraphs), recursing to finer ones
    (lines, sentences, words) only for pieces that are still too large. Pieces are then
    merged back greedily up to `chunk_size`, carrying `overlap` characters of trailing
    context into the next chunk.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 150,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        validate_sizes(chunk_size, overlap)
        self._size = chunk_size
        self._overlap = overlap
        self._separators = tuple(separators)

    def split(self, text: str) -> list[Chunk]:
        pieces = self._split(text, self._separators)
        merged = self._merge(pieces)
        cleaned = [chunk.strip() for chunk in merged]
        return [Chunk(index=i, text=t) for i, t in enumerate(c for c in cleaned if c)]

    def _split(self, text: str, separators: Sequence[str]) -> list[str]:
        if len(text) <= self._size:
            return [text]
        if not separators:
            return [text[i : i + self._size] for i in range(0, len(text), self._size)]
        head, *tail = separators
        pieces: list[str] = []
        for part in self._split_keep_separator(text, head):
            pieces.extend(self._split(part, tail) if len(part) > self._size else [part])
        return pieces

    @staticmethod
    def _split_keep_separator(text: str, separator: str) -> list[str]:
        parts = text.split(separator)
        pieces = [part + separator for part in parts[:-1]] + [parts[-1]]
        return [piece for piece in pieces if piece]

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        length = 0
        for piece in pieces:
            if current and length + len(piece) > self._size:
                chunks.append("".join(current))
                current, length = self._carry_overlap(current)
                while current and length + len(piece) > self._size:
                    length -= len(current.pop(0))
            current.append(piece)
            length += len(piece)
        if current:
            chunks.append("".join(current))
        return chunks

    def _carry_overlap(self, pieces: list[str]) -> tuple[list[str], int]:
        carried: list[str] = []
        total = 0
        for piece in reversed(pieces):
            if total + len(piece) > self._overlap:
                break
            carried.insert(0, piece)
            total += len(piece)
        return carried, total
