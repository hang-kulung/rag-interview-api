from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.booking import BookingResponse, TurnIntent


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w\-:.]+$")
    message: str = Field(min_length=1, max_length=4000)


class SourceChunk(BaseModel):
    document_id: UUID
    filename: str
    chunk_index: int
    text: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: TurnIntent
    sources: list[SourceChunk] = Field(default_factory=list)
    booking: BookingResponse | None = None
