"""Service-level flow tests using fakes for Gemini/Qdrant, real SQLite and fakeredis."""

import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.enums import ChunkingStrategy, DocumentStatus
from app.core.exceptions import EmbeddingError, UnsupportedFileTypeError
from app.db import build_engine, build_sessionmaker, init_models
from app.repositories.booking_repo import BookingRepository
from app.repositories.document_repo import DocumentRepository
from app.schemas.booking import BookingTurnAnalysis, TurnIntent
from app.schemas.chat import ChatMessage, SourceChunk
from app.services.booking import BookingService
from app.services.chat import ChatService
from app.services.chunking import Chunk
from app.services.ingestion import IngestionService
from app.services.llm import T
from app.services.memory import RedisBookingStateStore, RedisChatMemory
from app.services.rag import RagService


class FakeLLM:
    def __init__(self, analyses: list[BookingTurnAnalysis]) -> None:
        self._analyses = iter(analyses)

    async def generate(self, *, system: str, messages: Sequence[ChatMessage]) -> str:
        return "Remote work is allowed two days a week."

    async def generate_structured(
        self, *, system: str, messages: Sequence[ChatMessage], schema: type[T]
    ) -> T:
        return cast(T, next(self._analyses))


class FakeEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self._fail:
            raise EmbeddingError("boom")
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list[Chunk] = []
        self.deleted: list[uuid.UUID] = []

    async def ensure_collection(self) -> None: ...

    async def upsert_chunks(
        self,
        *,
        document_id: uuid.UUID,
        filename: str,
        strategy: ChunkingStrategy,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self.upserted.extend(chunks)

    async def search(
        self, vector: Sequence[float], *, top_k: int, score_threshold: float | None = None
    ) -> list[SourceChunk]:
        return [
            SourceChunk(
                document_id=uuid.uuid4(),
                filename="handbook.txt",
                chunk_index=0,
                text="Employees may work remotely two days per week.",
                score=0.9,
            )
        ]

    async def delete_document(self, document_id: uuid.UUID) -> None:
        self.deleted.append(document_id)


@pytest_asyncio.fixture
async def sessionmaker(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = build_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await init_models(engine)
    yield build_sessionmaker(engine)
    await engine.dispose()


async def test_multi_turn_booking_then_question(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    future = (date.today() + timedelta(days=5)).isoformat()
    llm = FakeLLM(
        [
            BookingTurnAnalysis(intent=TurnIntent.BOOKING, name="Anna Lee"),
            BookingTurnAnalysis(
                intent=TurnIntent.BOOKING,
                email="anna@example.com",
                interview_date=future,
                interview_time="10:00",
            ),
            BookingTurnAnalysis(intent=TurnIntent.QUESTION),
        ]
    )
    async with sessionmaker() as session:
        memory = RedisChatMemory(redis, ttl_seconds=60, max_messages=20)
        state_store = RedisBookingStateStore(redis, ttl_seconds=60)
        service = ChatService(
            memory=memory,
            booking_state=state_store,
            rag=RagService(
                embedder=FakeEmbedder(),
                vector_store=FakeVectorStore(),
                llm=llm,
                top_k=3,
                score_threshold=None,
            ),
            booking=BookingService(llm=llm, repository=BookingRepository(session)),
        )

        first = await service.handle(session_id="s1", message="Book me an interview, I'm Anna Lee")
        assert first.booking is None
        assert "email address" in first.answer
        assert await state_store.get("s1") is not None  # partial state kept between turns

        second = await service.handle(session_id="s1", message="anna@example.com, in 5 days, 10am")
        assert second.booking is not None
        assert second.booking.email == "anna@example.com"
        assert await state_store.get("s1") is None  # cleared after persisting

        third = await service.handle(session_id="s1", message="What is the remote work policy?")
        assert third.intent is TurnIntent.QUESTION
        assert third.sources and third.sources[0].filename == "handbook.txt"

        assert len(await memory.get_history("s1")) == 6
        assert len(await BookingRepository(session).list_for_session("s1")) == 1


async def test_ingestion_success_and_failure_paths(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    text = ("Employees may work remotely two days per week. " * 60).encode()
    async with sessionmaker() as session:
        store = FakeVectorStore()
        service = IngestionService(
            documents=DocumentRepository(session),
            embedder=FakeEmbedder(),
            vector_store=store,
            max_upload_bytes=1_000_000,
            chunk_size=500,
            chunk_overlap=50,
        )
        document = await service.ingest(
            filename="handbook.txt",
            content_type="text/plain",
            content=text,
            strategy=ChunkingStrategy.FIXED,
        )
        assert document.status is DocumentStatus.READY
        assert document.chunk_count == len(store.upserted) > 1

        with pytest.raises(UnsupportedFileTypeError):
            await service.ingest(
                filename="notes.docx",
                content_type=None,
                content=b"x",
                strategy=ChunkingStrategy.FIXED,
            )

        failing_store = FakeVectorStore()
        failing = IngestionService(
            documents=DocumentRepository(session),
            embedder=FakeEmbedder(fail=True),
            vector_store=failing_store,
            max_upload_bytes=1_000_000,
            chunk_size=500,
            chunk_overlap=50,
        )
        with pytest.raises(EmbeddingError):
            await failing.ingest(
                filename="handbook.txt",
                content_type="text/plain",
                content=text,
                strategy=ChunkingStrategy.RECURSIVE,
            )
        assert len(failing_store.deleted) == 1  # vectors cleaned up
