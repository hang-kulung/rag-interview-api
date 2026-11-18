from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile, status

from app.api.deps import IngestionServiceDep
from app.core.enums import ChunkingStrategy
from app.schemas.ingestion import DocumentResponse

router = APIRouter(prefix="/documents", tags=["ingestion"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    service: IngestionServiceDep,
    file: Annotated[UploadFile, File(description="A .pdf or .txt file")],
    chunking_strategy: Annotated[
        ChunkingStrategy, Form(description="Chunking strategy to apply")
    ] = ChunkingStrategy.RECURSIVE,
) -> DocumentResponse:
    content = await file.read()
    document = await service.ingest(
        filename=file.filename or "unnamed",
        content_type=file.content_type,
        content=content,
        strategy=chunking_strategy,
    )
    return DocumentResponse.model_validate(document)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: UUID, service: IngestionServiceDep) -> DocumentResponse:
    return DocumentResponse.model_validate(await service.get(document_id))
