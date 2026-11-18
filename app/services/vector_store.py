"""Step 3: VectorStore Protocol + Qdrant implementation."""

import uuid
from collections.abc import Sequence
from typing import Protocol

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ApiException

from app.core.enums import ChunkingStrategy
from app.core.exceptions import VectorStoreError
from app.schemas.chat import SourceChunk
from app.services.chunking import Chunk


class VectorStore(Protocol):
    async def ensure_collection(self) -> None: ...

    async def upsert_chunks(
        self,
        *,
        document_id: uuid.UUID,
        filename: str,
        strategy: ChunkingStrategy,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    async def search(
        self, vector: Sequence[float], *, top_k: int, score_threshold: float | None = None
    ) -> list[SourceChunk]: ...

    async def delete_document(self, document_id: uuid.UUID) -> None: ...


class QdrantVectorStore:
    _UPSERT_BATCH = 128

    def __init__(self, client: AsyncQdrantClient, *, collection: str, dimensions: int) -> None:
        self._client = client
        self._collection = collection
        self._dimensions = dimensions

    async def ensure_collection(self) -> None:
        try:
            if await self._client.collection_exists(self._collection):
                return
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimensions, distance=models.Distance.COSINE
                ),
            )
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except (ApiException, ConnectionError) as exc:
            raise VectorStoreError(f"Could not prepare Qdrant collection: {exc}") from exc

    async def upsert_chunks(
        self,
        *,
        document_id: uuid.UUID,
        filename: str,
        strategy: ChunkingStrategy,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=list(vector),
                payload={
                    "document_id": str(document_id),
                    "filename": filename,
                    "chunk_index": chunk.index,
                    "strategy": strategy.value,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            for start in range(0, len(points), self._UPSERT_BATCH):
                await self._client.upsert(
                    collection_name=self._collection,
                    points=points[start : start + self._UPSERT_BATCH],
                    wait=True,
                )
        except (ApiException, ConnectionError) as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    async def search(
        self, vector: Sequence[float], *, top_k: int, score_threshold: float | None = None
    ) -> list[SourceChunk]:
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except (ApiException, ConnectionError) as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc
        return [
            SourceChunk(
                document_id=uuid.UUID(str(point.payload["document_id"])),
                filename=str(point.payload["filename"]),
                chunk_index=int(point.payload["chunk_index"]),
                text=str(point.payload["text"]),
                score=point.score,
            )
            for point in response.points
            if point.payload is not None
        ]

    async def delete_document(self, document_id: uuid.UUID) -> None:
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=str(document_id)),
                            )
                        ]
                    )
                ),
            )
        except (ApiException, ConnectionError) as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc
