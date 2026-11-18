from enum import StrEnum


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


class FileType(StrEnum):
    PDF = "pdf"
    TXT = "txt"


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
