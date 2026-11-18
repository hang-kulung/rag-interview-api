from __future__ import annotations

from datetime import date, datetime, time
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class TurnIntent(StrEnum):
    QUESTION = "question"
    BOOKING = "booking"
    CANCEL_BOOKING = "cancel_booking"


class BookingTurnAnalysis(BaseModel):
    """Raw LLM output for one chat turn. Values are unvalidated strings by design."""

    intent: TurnIntent
    name: str | None = None
    email: str | None = None
    interview_date: str | None = Field(default=None, description="ISO date, YYYY-MM-DD")
    interview_time: str | None = Field(default=None, description="24-hour time, HH:MM")


class BookingState(BaseModel):
    """Booking slots collected so far in a session (partial, unvalidated)."""

    name: str | None = None
    email: str | None = None
    interview_date: str | None = None
    interview_time: str | None = None

    def merged_with(self, analysis: BookingTurnAnalysis) -> BookingState:
        """Newly extracted values override older ones; missing values never erase data."""
        return BookingState(
            name=analysis.name or self.name,
            email=analysis.email or self.email,
            interview_date=analysis.interview_date or self.interview_date,
            interview_time=analysis.interview_time or self.interview_time,
        )

    def without(self, fields: list[str]) -> BookingState:
        return BookingState.model_validate(
            {key: (None if key in fields else value) for key, value in self.model_dump().items()}
        )

    def is_empty(self) -> bool:
        return not any(self.model_dump().values())


class BookingDetails(BaseModel):
    """A fully validated booking, ready to persist."""

    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    interview_date: date
    interview_time: time

    @field_validator("interview_date")
    @classmethod
    def _date_not_in_past(cls, value: date) -> date:
        if value < date.today():
            raise ValueError("interview date cannot be in the past")
        return value


class BookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: str
    name: str
    email: str
    interview_date: date
    interview_time: time
    created_at: datetime
