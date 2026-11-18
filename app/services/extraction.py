import io
from pathlib import PurePath
from typing import Final

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.core.enums import FileType
from app.core.exceptions import EmptyDocumentError, ExtractionError, UnsupportedFileTypeError

_EXTENSIONS: Final[dict[str, FileType]] = {".pdf": FileType.PDF, ".txt": FileType.TXT}
_CONTENT_TYPES: Final[dict[FileType, set[str]]] = {
    FileType.PDF: {"application/pdf", "application/octet-stream"},
    FileType.TXT: {"text/plain", "application/octet-stream"},
}


def detect_file_type(filename: str, content_type: str | None) -> FileType:
    """Resolve the file type from the extension, cross-checked against the MIME type."""
    file_type = _EXTENSIONS.get(PurePath(filename).suffix.lower())
    if file_type is None:
        raise UnsupportedFileTypeError("Only .pdf and .txt files are supported.")
    if content_type and content_type.split(";")[0].strip() not in _CONTENT_TYPES[file_type]:
        raise UnsupportedFileTypeError(
            f"Content type '{content_type}' does not match a .{file_type.value} file."
        )
    return file_type


def extract_text(content: bytes, file_type: FileType) -> str:
    """Extract plain text from raw file bytes. CPU-bound: run in a worker thread."""
    text = _extract_pdf(content) if file_type is FileType.PDF else _extract_txt(content)
    if not text.strip():
        raise EmptyDocumentError("No extractable text found in the document.")
    return text


def _extract_pdf(content: bytes) -> str:
    if not content.startswith(b"%PDF-"):
        raise ExtractionError("File is not a valid PDF.")
    try:
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except PyPdfError as exc:
        raise ExtractionError(f"Could not read PDF: {exc}") from exc


def _extract_txt(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")
