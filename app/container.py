from dataclasses import dataclass

from google import genai
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db import build_engine, build_sessionmaker
from app.services.embeddings import EmbeddingClient, GeminiEmbeddingClient
from app.services.llm import GeminiLLMClient, LLMClient
from app.services.memory import RedisChatMemory, RedisBookingStateStore
from app.services.vector_store import QdrantVectorStore, VectorStore


@dataclass(slots=True)
class Container:
    """Long-lived, shared resources created once at startup and closed at shutdown."""

    settings: Settings
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    redis: Redis
    qdrant: AsyncQdrantClient
    embedder: EmbeddingClient
    llm: LLMClient
    vector_store: VectorStore
    memory: RedisChatMemory
    booking_state: RedisBookingStateStore

    async def aclose(self) -> None:
        await self.redis.aclose()
        await self.qdrant.close()
        await self.engine.dispose()


def build_container(settings: Settings) -> Container:
    engine = build_engine(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
    )
    gemini = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    return Container(
        settings=settings,
        engine=engine,
        sessionmaker=build_sessionmaker(engine),
        redis=redis,
        qdrant=qdrant,
        embedder=GeminiEmbeddingClient(
            gemini,
            model=settings.gemini_embedding_model,
            dimensions=settings.embedding_dimensions,
        ),
        llm=GeminiLLMClient(gemini, model=settings.gemini_chat_model),
        vector_store=QdrantVectorStore(
            qdrant,
            collection=settings.qdrant_collection,
            dimensions=settings.embedding_dimensions,
        ),
        memory=RedisChatMemory(
            redis,
            ttl_seconds=settings.chat_ttl_seconds,
            max_messages=settings.chat_history_max_messages,
        ),
        booking_state=RedisBookingStateStore(
            redis,
            ttl_seconds=settings.chat_ttl_seconds,
        ),
    )
