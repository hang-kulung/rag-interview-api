import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import ChunkingStrategy, DocumentStatus, FileType
from app.models.document import Document


class DocumentRepository:
    """Persistence for document metadata. Each write method commits."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, filename: str, file_type: FileType, strategy: ChunkingStrategy
    ) -> Document:
        document = Document(
            filename=filename,
            file_type=file_type,
            chunking_strategy=strategy,
            status=DocumentStatus.PROCESSING,
        )
        self._session.add(document)
        await self._session.commit()
        return document

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self._session.get(Document, document_id)

    async def mark_ready(self, document: Document, *, chunk_count: int) -> Document:
        document.status = DocumentStatus.READY
        document.chunk_count = chunk_count
        await self._session.commit()
        return document

    async def mark_failed(self, document: Document, error: str) -> Document:
        document.status = DocumentStatus.FAILED
        document.error = error[:1000]
        await self._session.commit()
        return document