# RAG Interview API

Backend with two REST APIs (FastAPI): document ingestion and conversational RAG with interview booking.

Stack: FastAPI, Qdrant, PostgreSQL (SQLAlchemy 2.0 async), Redis, Gemini (LLM + embeddings).

## Build order

Each stub file starts with a docstring naming the step that fills it in.

| Step | Work | Files |
|---|---|---|
| 2 | Config, DB, models, repositories | `core/`, `db.py`, `models/`, `repositories/` |
| 3 | Ingestion API | `schemas/ingestion.py`, `services/chunking/`, `extraction.py`, `embeddings.py`, `vector_store.py`, `ingestion.py`, `routes/ingestion.py` |
| 4 | Conversational RAG API | `schemas/chat.py`, `llm.py`, `memory.py`, `rag.py`, `routes/chat.py` |
| 5 | Interview booking | `schemas/booking.py`, `services/booking.py`, `services/chat.py` |
| 6 | Typing and error handling | whole codebase (`mypy --strict`, `ruff`) |
| 7 | Tests | `tests/` |
| 8 | Docs and delivery | this README |

Wiring files (`container.py`, `api/deps.py`, `main.py`, `routes/health.py`) are filled in as the steps need them.

## Infrastructure

```bash
cp .env.example .env     # set GEMINI_API_KEY
docker compose up -d postgres redis qdrant
```
