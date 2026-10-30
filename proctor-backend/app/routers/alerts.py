import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Alert, AlertStatus, BehaviorType, ExamSession
from ..schemas import AlertCreate, AlertOut, ReviewIn
from ..security import require_detector_key

router = APIRouter(prefix="/alerts", tags=["alerts"])

ALLOWED_CLIP_EXT = {".mp4", ".webm"}


def _to_out(a: Alert) -> AlertOut:
    out = AlertOut.model_validate(a)
    out.clip_url = f"/clips/{Path(a.clip_path).name}" if a.clip_path else None
    return out


def _get_or_404(db: Session, alert_id: int) -> Alert:
    obj = db.get(Alert, alert_id)
    if not obj:
        raise HTTPException(404, "Alert not found")
    return obj


@router.post("", response_model=AlertOut, status_code=201, dependencies=[Depends(require_detector_key)])
def create_alert(body: AlertCreate, db: Session = Depends(get_db)):
    """Called by the detection worker when it flags something."""
    if not db.get(ExamSession, body.session_id):
        raise HTTPException(404, "Session not found")
    data = body.model_dump()
    data["detected_at"] = data["detected_at"] or datetime.now(timezone.utc)
    obj = Alert(**data)
    db.add(obj)
    db.commit()
    return _to_out(obj)


@router.post("/{alert_id}/clip", response_model=AlertOut, dependencies=[Depends(require_detector_key)])
def upload_clip(alert_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Attach the short evidence video clip to an alert."""
    obj = _get_or_404(db, alert_id)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_CLIP_EXT:
        raise HTTPException(400, f"Clip must be one of {sorted(ALLOWED_CLIP_EXT)}")
    dest_dir = Path(settings.clips_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"alert_{alert_id}{ext}"  # server-chosen name, never the client's
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    obj.clip_path = str(dest)
    db.commit()
    return _to_out(obj)


@router.get("", response_model=list[AlertOut])
def list_alerts(
    session_id: int | None = None,
    status: AlertStatus | None = None,
    behavior_type: BehaviorType | None = None,
    min_confidence: float = Query(0, ge=0, le=1),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = select(Alert).where(Alert.confidence >= min_confidence)
    if session_id is not None:
        q = q.where(Alert.session_id == session_id)
    if status:
        q = q.where(Alert.status == status)
    if behavior_type:
        q = q.where(Alert.behavior_type == behavior_type)
    q = q.order_by(Alert.detected_at.desc()).limit(limit).offset(offset)
    return [_to_out(a) for a in db.scalars(q).all()]


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    return _to_out(_get_or_404(db, alert_id))


@router.patch("/{alert_id}/review", response_model=AlertOut)
def review_alert(alert_id: int, body: ReviewIn, db: Session = Depends(get_db)):
    """Invigilator confirms or dismisses a flag. The human makes the final call."""
    if body.status == AlertStatus.pending:
        raise HTTPException(400, "Review status must be 'confirmed' or 'dismissed'")
    obj = _get_or_404(db, alert_id)
    obj.status = body.status
    obj.reviewed_by = body.reviewed_by
    obj.review_notes = body.notes
    obj.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    return _to_out(obj)
