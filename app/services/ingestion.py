import asyncio
import logging
import uuid

from app.core.enums import ChunkingStrategy
from app.core.exceptions import EmptyDocumentError, FileTooLargeError, NotFoundError
from app.models.document import Document
from app.repositories.document_repo import DocumentRepository
from app.services.chunking import get_chunker
from app.services.embeddings import EmbeddingClient
from app.services.extraction import detect_file_type, extract_text
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    """Upload -> extract -> chunk -> embed -> index -> record metadata."""

    def __init__(
        self,
        *,
        documents: DocumentRepository,
        embedder: EmbeddingClient,
        vector_store: VectorStore,
        max_upload_bytes: int,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._documents = documents
        self._embedder = embedder
        self._vector_store = vector_store
        self._max_upload_bytes = max_upload_bytes
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
        strategy: ChunkingStrategy,
    ) -> Document:
        file_type = detect_file_type(filename, content_type)
        if len(content) > self._max_upload_bytes:
            raise FileTooLargeError(
                f"File exceeds the {self._max_upload_bytes // (1024 * 1024)} MB upload limit."
            )

        text = await asyncio.to_thread(extract_text, content, file_type)
        chunker = get_chunker(strategy, chunk_size=self._chunk_size, overlap=self._chunk_overlap)
        chunks = chunker.split(text)
        if not chunks:
            raise EmptyDocumentError("The document produced no chunks.")

        document = await self._documents.create(
            filename=filename, file_type=file_type, strategy=strategy
        )
        try:
            vectors = await self._embedder.embed_documents([chunk.text for chunk in chunks])
            await self._vector_store.upsert_chunks(
                document_id=document.id,
                filename=filename,
                strategy=strategy,
                chunks=chunks,
                vectors=vectors,
            )
        except Exception as exc:
            logger.exception("Ingestion failed for document %s", document.id)
            await self._cleanup_vectors(document.id)
            await self._documents.mark_failed(document, str(exc))
            raise
        return await self._documents.mark_ready(document, chunk_count=len(chunks))

    async def get(self, document_id: uuid.UUID) -> Document:
        document = await self._documents.get(document_id)
        if document is None:
            raise NotFoundError(f"Document {document_id} not found.")
        return document

    async def _cleanup_vectors(self, document_id: uuid.UUID) -> None:
        try:
            await self._vector_store.delete_document(document_id)
        except Exception:
            logger.warning("Could not clean up vectors for document %s", document_id)
