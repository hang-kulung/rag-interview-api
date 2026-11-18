from collections.abc import Sequence
from typing import Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.exceptions import EmbeddingError


class EmbeddingClient(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbeddingClient:
    """Gemini embeddings with task-specific types for documents vs. queries.

    Vectors are compared with cosine distance in the vector store, so reduced-dimension
    outputs need no extra normalisation.
    """

    _MAX_BATCH = 100  # Gemini's per-request input limit

    def __init__(self, client: genai.Client, *, model: str, dimensions: int) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text], task_type="RETRIEVAL_QUERY")
        return vectors[0]

    async def _embed(self, texts: Sequence[str], *, task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._MAX_BATCH):
            batch = list(texts[start : start + self._MAX_BATCH])
            try:
                response = await self._client.aio.models.embed_content(
                    model=self._model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=self._dimensions
                    ),
                )
            except genai_errors.APIError as exc:
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc
            vectors.extend(list(item.values or []) for item in response.embeddings or [])
        if len(vectors) != len(texts):
            raise EmbeddingError("Embedding service returned an unexpected number of vectors.")
        return vectors
