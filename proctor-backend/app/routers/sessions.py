from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Alert, AlertStatus, ExamSession
from ..schemas import SessionCreate, SessionOut, SessionStats

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_or_404(db: Session, session_id: int) -> ExamSession:
    obj = db.get(ExamSession, session_id)
    if not obj:
        raise HTTPException(404, "Session not found")
    return obj


@router.post("", response_model=SessionOut, status_code=201)
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    obj = ExamSession(**body.model_dump())
    db.add(obj)
    db.commit()
    return obj


@router.get("", response_model=list[SessionOut])
def list_sessions(db: Session = Depends(get_db)):
    return db.scalars(select(ExamSession).order_by(ExamSession.started_at.desc())).all()


@router.get("/{session_id}", response_model=SessionOut)
def get_session(session_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, session_id)


@router.post("/{session_id}/end", response_model=SessionOut)
def end_session(session_id: int, db: Session = Depends(get_db)):
    obj = _get_or_404(db, session_id)
    obj.ended_at = datetime.now(timezone.utc)
    db.commit()
    return obj


@router.get("/{session_id}/stats", response_model=SessionStats)
def session_stats(session_id: int, db: Session = Depends(get_db)):
    _get_or_404(db, session_id)
    by_behavior = dict(
        db.execute(
            select(Alert.behavior_type, func.count()).where(Alert.session_id == session_id).group_by(Alert.behavior_type)
        ).all()
    )
    by_status = dict(
        db.execute(
            select(Alert.status, func.count()).where(Alert.session_id == session_id).group_by(Alert.status)
        ).all()
    )
    by_behavior = {k.value: v for k, v in by_behavior.items()}
    by_status = {k.value: v for k, v in by_status.items()}
    confirmed = by_status.get(AlertStatus.confirmed.value, 0)
    dismissed = by_status.get(AlertStatus.dismissed.value, 0)
    reviewed = confirmed + dismissed
    return SessionStats(
        session_id=session_id,
        total=sum(by_status.values()),
        by_behavior=by_behavior,
        by_status=by_status,
        precision_so_far=(confirmed / reviewed) if reviewed else None,
    )
