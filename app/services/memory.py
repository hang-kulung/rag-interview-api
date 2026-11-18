from collections.abc import Sequence

from redis.asyncio import Redis

from app.schemas.chat import ChatMessage
from app.schemas.booking import BookingState


class RedisChatMemory:
    """Per-session conversation history, stored as a capped Redis list with a TTL."""

    def __init__(self, redis: Redis, *, ttl_seconds: int, max_messages: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._max_messages = max_messages

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:{session_id}"

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        raw_items = await self._redis.lrange(self._key(session_id), 0, -1)
        return [ChatMessage.model_validate_json(item) for item in raw_items]

    async def append(self, session_id: str, messages: Sequence[ChatMessage]) -> None:
        key = self._key(session_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *(message.model_dump_json() for message in messages))
            pipe.ltrim(key, -self._max_messages, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))


class RedisBookingStateStore:
    """Partially collected booking details for a session (slot filling across turns)."""

    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"booking:{session_id}"

    async def get(self, session_id: str) -> BookingState | None:
        raw = await self._redis.get(self._key(session_id))
        return BookingState.model_validate_json(raw) if raw else None

    async def save(self, session_id: str, state: BookingState) -> None:
        await self._redis.set(self._key(session_id), state.model_dump_json(), ex=self._ttl)

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
