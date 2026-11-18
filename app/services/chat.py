from datetime import date
from typing import Final

from app.schemas.booking import (
    BookingResponse,
    BookingState,
    BookingTurnAnalysis,
    TurnIntent,
)
from app.schemas.chat import ChatMessage, ChatResponse, Role, SourceChunk
from app.services.booking import BookingCheck, BookingService
from app.services.memory import RedisBookingStateStore, RedisChatMemory
from app.services.rag import RagService

FIELD_LABELS: Final[dict[str, str]] = {
    "name": "full name",
    "email": "email address",
    "interview_date": "preferred interview date",
    "interview_time": "preferred interview time",
}

CANCEL_REPLY: Final = (
    "No problem, I've cancelled the booking request. Is there anything else I can help with?"
)


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"


class ChatService:
    """Orchestrates one chat turn: memory -> intent -> (booking | RAG) -> memory."""

    def __init__(
        self,
        *,
        memory: RedisChatMemory,
        booking_state: RedisBookingStateStore,
        rag: RagService,
        booking: BookingService,
    ) -> None:
        self._memory = memory
        self._booking_state = booking_state
        self._rag = rag
        self._booking = booking

    async def handle(self, *, session_id: str, message: str) -> ChatResponse:
        history = await self._memory.get_history(session_id)
        state = await self._booking_state.get(session_id) or BookingState()
        analysis = await self._booking.analyze_turn(
            message=message, history=history, state=state, today=date.today()
        )

        sources: list[SourceChunk] = []
        booking: BookingResponse | None = None

        if analysis.intent is TurnIntent.CANCEL_BOOKING:
            await self._booking_state.clear(session_id)
            answer = CANCEL_REPLY
        elif analysis.intent is TurnIntent.BOOKING:
            answer, booking = await self._continue_booking(session_id, state, analysis)
        else:
            result = await self._rag.answer(question=message, history=history)
            answer, sources = result.answer, result.sources

        await self._memory.append(
            session_id,
            [
                ChatMessage(role=Role.USER, content=message),
                ChatMessage(role=Role.ASSISTANT, content=answer),
            ],
        )
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            intent=analysis.intent,
            sources=sources,
            booking=booking,
        )

    async def _continue_booking(
        self, session_id: str, state: BookingState, analysis: BookingTurnAnalysis
    ) -> tuple[str, BookingResponse | None]:
        merged = state.merged_with(analysis)
        check = self._booking.check(merged)

        if check.details is not None:
            saved = await self._booking.save(session_id=session_id, details=check.details)
            await self._booking_state.clear(session_id)
            reply = (
                f"You're all set, {check.details.name}! Your interview is booked for "
                f"{check.details.interview_date:%A, %d %B %Y} at "
                f"{check.details.interview_time:%H:%M}, under {check.details.email}."
            )
            return reply, BookingResponse.model_validate(saved)

        # Drop invalid values so the user can re-supply them; keep everything else.
        await self._booking_state.save(session_id, merged.without(list(check.invalid)))
        return self._follow_up(check), None

    @staticmethod
    def _follow_up(check: BookingCheck) -> str:
        parts: list[str] = []
        if check.invalid:
            problems = "; ".join(
                f"the {FIELD_LABELS[name]} ({message})" for name, message in check.invalid.items()
            )
            parts.append(f"I couldn't use {problems}.")
        needed = [*check.invalid, *check.missing]
        if needed:
            labels = _join([FIELD_LABELS[name] for name in needed])
            parts.append(f"To book your interview, please share your {labels}.")
        return " ".join(parts)
