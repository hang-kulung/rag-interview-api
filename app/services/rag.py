from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.chat import ChatMessage, Role, SourceChunk
from app.services.embeddings import EmbeddingClient
from app.services.llm import LLMClient
from app.services.vector_store import VectorStore

REWRITE_SYSTEM_PROMPT = (
    "You rewrite follow-up questions into standalone search queries. Using the conversation, "
    "rewrite the final user question so it is fully self-contained (resolve pronouns and "
    "references). Return ONLY the rewritten question, with no commentary."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context "
    "excerpts. If the context does not contain the answer, say you don't know based on the "
    "uploaded documents. Be concise and do not invent facts. Mention the source filename "
    "when it helps."
)

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant in the uploaded documents. "
    "Could you rephrase, or upload a document that covers this?"
)

_REWRITE_HISTORY_WINDOW = 6


@dataclass(frozen=True, slots=True)
class RagResult:
    answer: str
    sources: list[SourceChunk]


class RagService:
    """Hand-rolled retrieval-augmented generation: rewrite -> retrieve -> prompt -> generate."""

    def __init__(
        self,
        *,
        embedder: EmbeddingClient,
        vector_store: VectorStore,
        llm: LLMClient,
        top_k: int,
        score_threshold: float | None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
        self._top_k = top_k
        self._score_threshold = score_threshold

    async def answer(self, *, question: str, history: Sequence[ChatMessage]) -> RagResult:
        standalone = await self._rewrite_query(question, history)
        query_vector = await self._embedder.embed_query(standalone)
        sources = await self._vector_store.search(
            query_vector, top_k=self._top_k, score_threshold=self._score_threshold
        )
        if not sources:
            return RagResult(answer=NO_CONTEXT_ANSWER, sources=[])

        grounded_question = self._build_grounded_prompt(question, sources)
        answer = await self._llm.generate(
            system=ANSWER_SYSTEM_PROMPT,
            messages=[*history, ChatMessage(role=Role.USER, content=grounded_question)],
        )
        return RagResult(answer=answer.strip(), sources=sources)

    async def _rewrite_query(self, question: str, history: Sequence[ChatMessage]) -> str:
        if not history:
            return question
        transcript = "\n".join(
            f"{message.role.value}: {message.content}"
            for message in history[-_REWRITE_HISTORY_WINDOW:]
        )
        prompt = f"Conversation so far:\n{transcript}\n\nFinal user question: {question}"
        rewritten = await self._llm.generate(
            system=REWRITE_SYSTEM_PROMPT,
            messages=[ChatMessage(role=Role.USER, content=prompt)],
        )
        return rewritten.strip() or question

    @staticmethod
    def _build_grounded_prompt(question: str, sources: Sequence[SourceChunk]) -> str:
        context = "\n\n".join(
            f"[{position}] (source: {source.filename}, chunk {source.chunk_index})\n{source.text}"
            for position, source in enumerate(sources, start=1)
        )
        return f"Context:\n{context}\n\nQuestion: {question}"
