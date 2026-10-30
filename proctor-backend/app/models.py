import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BehaviorType(str, enum.Enum):
    phone = "phone"
    looking_at_neighbor = "looking_at_neighbor"
    talking = "talking"
    passing_item = "passing_item"
    leaving_seat = "leaving_seat"
    other = "other"


class AlertStatus(str, enum.Enum):
    pending = "pending"
    confirmed = "confirmed"
    dismissed = "dismissed"


class ExamSession(Base):
    """One exam in one hall, monitored by one camera."""

    __tablename__ = "exam_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    hall: Mapped[str] = mapped_column(String(100))
    camera_id: Mapped[str] = mapped_column(String(100))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    alerts: Mapped[list["Alert"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Alert(Base):
    """A flag raised by the detector, awaiting human review."""

    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_session_status", "session_id", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("exam_sessions.id", ondelete="CASCADE"))
    seat_label: Mapped[str | None] = mapped_column(String(20))   # e.g. "B4"
    track_id: Mapped[int | None] = mapped_column(Integer)         # tracker ID from the detector
    behavior_type: Mapped[BehaviorType] = mapped_column(Enum(BehaviorType, name="behavior_type"))
    confidence: Mapped[float] = mapped_column(Float)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    video_ts_seconds: Mapped[float | None] = mapped_column(Float)
    clip_path: Mapped[str | None] = mapped_column(String(500))
    details: Mapped[dict | None] = mapped_column(JSONB)          # raw model output: head angles, bbox, etc.

    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, name="alert_status"), default=AlertStatus.pending
    )
    reviewed_by: Mapped[str | None] = mapped_column(String(100))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_notes: Mapped[str | None] = mapped_column(Text)

    session: Mapped[ExamSession] = relationship(back_populates="alerts")
