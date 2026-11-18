from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    text: str


class Chunker(Protocol):
    """A chunking strategy: turns raw document text into ordered chunks."""

    def split(self, text: str) -> list[Chunk]: ...


def validate_sizes(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
