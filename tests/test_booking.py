from datetime import date, timedelta

from app.schemas.booking import BookingState, BookingTurnAnalysis, TurnIntent
from app.services.booking import BookingService


def _future(days: int = 3) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def test_check_reports_missing_fields() -> None:
    check = BookingService.check(BookingState(name="Anna Lee"))
    assert check.details is None
    assert check.missing == ["email", "interview_date", "interview_time"]


def test_check_accepts_complete_valid_booking() -> None:
    state = BookingState(
        name="Anna Lee",
        email="anna@example.com",
        interview_date=_future(),
        interview_time="14:30",
    )
    check = BookingService.check(state)
    assert check.details is not None
    assert check.details.interview_time.hour == 14
    assert not check.missing and not check.invalid


def test_check_flags_invalid_email_and_past_date() -> None:
    state = BookingState(
        name="Anna Lee",
        email="not-an-email",
        interview_date="2020-01-01",
        interview_time="14:30",
    )
    check = BookingService.check(state)
    assert check.details is None
    assert set(check.invalid) == {"email", "interview_date"}


def test_merge_keeps_existing_values_when_new_turn_has_nulls() -> None:
    state = BookingState(name="Anna Lee", email="anna@example.com")
    analysis = BookingTurnAnalysis(intent=TurnIntent.BOOKING, interview_time="09:00")
    merged = state.merged_with(analysis)
    assert merged.name == "Anna Lee"
    assert merged.email == "anna@example.com"
    assert merged.interview_time == "09:00"


def test_without_clears_only_named_fields() -> None:
    state = BookingState(name="Anna Lee", email="bad")
    assert state.without(["email"]).email is None
    assert state.without(["email"]).name == "Anna Lee"
