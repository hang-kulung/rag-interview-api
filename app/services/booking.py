from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Final

from pydantic import ValidationError

from app.models.booking import Booking
from app.repositories.booking_repo import BookingRepository
from app.schemas.booking import BookingDetails, BookingState, BookingTurnAnalysis
from app.schemas.chat import ChatMessage, Role
from app.services.llm import LLMClient

REQUIRED_FIELDS: Final[tuple[str, ...]] = ("name", "email", "interview_date", "interview_time")

ANALYSIS_SYSTEM_PROMPT = """\
You analyse one turn of a conversation with an assistant that answers questions about uploaded
documents AND can book job interviews.

Today is {today} ({weekday}). Use it to resolve relative dates such as "tomorrow" or "next Friday".

Booking details collected so far in this conversation:
{state}

Classify the user's LATEST message as exactly one intent:
- "booking": the user wants to book/schedule an interview, OR is supplying booking details
  (name, email, date, time), OR is answering a question the assistant asked about a booking.
- "cancel_booking": the user wants to cancel / abandon the booking in progress.
- "question": anything else (a question about the documents, small talk, etc.).

Extract booking details ONLY if the user actually stated them in this conversation. Never
guess or invent a value; use null for anything not stated.
- interview_date: ISO format YYYY-MM-DD
- interview_time: 24-hour format HH:MM
"""


@dataclass(frozen=True, slots=True)
class BookingCheck:
    """Outcome of validating the collected booking slots."""

    details: BookingDetails | None = None
    missing: list[str] = field(default_factory=list)
    invalid: dict[str, str] = field(default_factory=dict)


class BookingService:
    def __init__(self, *, llm: LLMClient, repository: BookingRepository) -> None:
        self._llm = llm
        self._repository = repository

    async def analyze_turn(
        self,
        *,
        message: str,
        history: Sequence[ChatMessage],
        state: BookingState,
        today: date,
    ) -> BookingTurnAnalysis:
        system = ANALYSIS_SYSTEM_PROMPT.format(
            today=today.isoformat(),
            weekday=today.strftime("%A"),
            state=state.model_dump_json(),
        )
        return await self._llm.generate_structured(
            system=system,
            messages=[*history[-6:], ChatMessage(role=Role.USER, content=message)],
            schema=BookingTurnAnalysis,
        )

    @staticmethod
    def check(state: BookingState) -> BookingCheck:
        values = state.model_dump()
        missing = [name for name in REQUIRED_FIELDS if not values[name]]
        if missing:
            return BookingCheck(missing=missing)
        try:
            return BookingCheck(details=BookingDetails.model_validate(values))
        except ValidationError as exc:
            invalid = {str(err["loc"][0]): err["msg"] for err in exc.errors() if err["loc"]}
            return BookingCheck(invalid=invalid)

    async def save(self, *, session_id: str, details: BookingDetails) -> Booking:
        return await self._repository.create(session_id=session_id, details=details)
