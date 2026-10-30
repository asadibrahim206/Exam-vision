from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import AlertStatus, BehaviorType


class SessionCreate(BaseModel):
    name: str
    hall: str
    camera_id: str


class SessionOut(SessionCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    started_at: datetime
    ended_at: datetime | None


class AlertCreate(BaseModel):
    session_id: int
    seat_label: str | None = None
    track_id: int | None = None
    behavior_type: BehaviorType
    confidence: float = Field(ge=0, le=1)
    detected_at: datetime | None = None
    video_ts_seconds: float | None = None
    details: dict[str, Any] | None = None


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    session_id: int
    seat_label: str | None
    track_id: int | None
    behavior_type: BehaviorType
    confidence: float
    detected_at: datetime
    video_ts_seconds: float | None
    clip_url: str | None = None
    details: dict[str, Any] | None
    status: AlertStatus
    reviewed_by: str | None
    reviewed_at: datetime | None
    review_notes: str | None


class ReviewIn(BaseModel):
    status: AlertStatus
    reviewed_by: str
    notes: str | None = None


class SessionStats(BaseModel):
    session_id: int
    total: int
    by_behavior: dict[str, int]
    by_status: dict[str, int]
    precision_so_far: float | None  # confirmed / (confirmed + dismissed), from human reviews
