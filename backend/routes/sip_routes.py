from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List
from .. import models
from ..database import get_db
from ..auth import get_current_user

router = APIRouter(prefix="/sips", tags=["sips"])

DAILY_GOAL_ML = 2000.0


class SipRequest(BaseModel):
    timestamp: float
    volume_ml: float
    sip_count: int


class DailyTotal(BaseModel):
    date: str
    total_ml: float
    sip_count: int


class StatsResponse(BaseModel):
    today_ml: float
    today_sips: int
    weekly_average_ml: float
    all_time_ml: float
    all_time_sips: int
    daily_goal_ml: float
    goal_percent: float


@router.post("")
def log_sip(
    body: SipRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    event = models.SipEvent(
        user_id=current_user.id,
        timestamp=body.timestamp,
        volume_ml=body.volume_ml,
        sip_count=body.sip_count
    )
    db.add(event)
    db.commit()
    return {"status": "ok"}


@router.get("/history", response_model=List[DailyTotal])
def get_history(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    since = datetime.utcnow() - timedelta(days=days)
    events = (
        db.query(models.SipEvent)
        .filter(
            models.SipEvent.user_id == current_user.id,
            models.SipEvent.recorded_at >= since
        )
        .order_by(models.SipEvent.recorded_at)
        .all()
    )

    # Group by date
    daily: dict = {}
    for event in events:
        date_str = event.recorded_at.strftime("%Y-%m-%d")
        if date_str not in daily:
            daily[date_str] = {"total_ml": 0.0, "sip_count": 0}
        daily[date_str]["total_ml"] += 15.0  # each event = one sip = 15ml
        daily[date_str]["sip_count"] += 1

    return [
        DailyTotal(date=date, total_ml=v["total_ml"], sip_count=v["sip_count"])
        for date, v in sorted(daily.items())
    ]


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now - timedelta(days=7)

    all_events = (
        db.query(models.SipEvent)
        .filter(models.SipEvent.user_id == current_user.id)
        .all()
    )

    today_events = [e for e in all_events if e.recorded_at >= today_start]
    week_events = [e for e in all_events if e.recorded_at >= week_start]

    today_sips = len(today_events)
    today_ml = today_sips * 15.0
    all_time_sips = len(all_events)
    all_time_ml = all_time_sips * 15.0

    # Weekly average: total ml this week divided by 7
    weekly_average_ml = (len(week_events) * 15.0) / 7

    goal_percent = min((today_ml / DAILY_GOAL_ML) * 100, 100)

    return StatsResponse(
        today_ml=today_ml,
        today_sips=today_sips,
        weekly_average_ml=round(weekly_average_ml, 1),
        all_time_ml=all_time_ml,
        all_time_sips=all_time_sips,
        daily_goal_ml=DAILY_GOAL_ML,
        goal_percent=round(goal_percent, 1)
    )
