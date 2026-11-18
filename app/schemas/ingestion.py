from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.core.enums import ChunkingStrategy, DocumentStatus, FileType


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    file_type: FileType
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    status: DocumentStatus
    error: str | None = None
    created_at: datetime
