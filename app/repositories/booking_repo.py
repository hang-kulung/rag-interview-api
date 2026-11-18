from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.booking import Booking
from app.schemas.booking import BookingDetails


class BookingRepository:
    """Persistence for interview bookings. Each write method commits."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, session_id: str, details: BookingDetails) -> Booking:
        booking = Booking(
            session_id=session_id,
            name=details.name,
            email=str(details.email),
            interview_date=details.interview_date,
            interview_time=details.interview_time,
        )
        self._session.add(booking)
        await self._session.commit()
        return booking

    async def list_for_session(self, session_id: str) -> list[Booking]:
        result = await self._session.execute(
            select(Booking).where(Booking.session_id == session_id).order_by(Booking.created_at)
        )
        return list(result.scalars().all())
