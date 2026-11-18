from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.container import Container
from app.repositories.document_repo import DocumentRepository
from app.repositories.booking_repo import BookingRepository
from app.services.booking import BookingService
from app.services.ingestion import IngestionService
from app.services.chat import ChatService
from app.services.rag import RagService


def get_container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


ContainerDep = Annotated[Container, Depends(get_container)]


async def get_db_session(container: ContainerDep) -> AsyncIterator[AsyncSession]:
    async with container.sessionmaker() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_chat_service(container: ContainerDep, session: SessionDep) -> ChatService:
    settings = container.settings
    return ChatService(
        memory=container.memory,
        booking_state=container.booking_state,
        rag=RagService(
            embedder=container.embedder,
            vector_store=container.vector_store,
            llm=container.llm,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        ),
        booking=BookingService(llm=container.llm, repository=BookingRepository(session)),
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


def get_ingestion_service(container: ContainerDep, session: SessionDep) -> IngestionService:
    settings = container.settings
    return IngestionService(
        documents=DocumentRepository(session),
        embedder=container.embedder,
        vector_store=container.vector_store,
        max_upload_bytes=settings.max_upload_bytes,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
