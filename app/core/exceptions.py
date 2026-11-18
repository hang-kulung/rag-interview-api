"""Step 2: AppError hierarchy + FastAPI exception handler."""

from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for errors that map cleanly to an HTTP response."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class EmptyDocumentError(AppError):
    status_code = 422
    code = "empty_document"


class ExtractionError(AppError):
    status_code = 422
    code = "extraction_failed"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class EmbeddingError(AppError):
    status_code = 502
    code = "embedding_failed"


class LLMError(AppError):
    status_code = 502
    code = "llm_failed"


class VectorStoreError(AppError):
    status_code = 502
    code = "vector_store_failed"


async def _handle_app_error(request: Request, exc: Exception) -> JSONResponse:
    error = cast(AppError, exc)
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": error.message}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
