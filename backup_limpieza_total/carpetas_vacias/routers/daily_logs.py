"""
Endpoints para la gestión de registros diarios.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from ..database import get_db
from ..models import DailyLog, Project
from ..schemas import DailyLogCreate, DailyLogUpdate, DailyLogResponse, ProjectResponse

router = APIRouter(
    prefix="/api/daily-logs",
    tags=["daily-logs"],
    responses={404: {"description": "No encontrado"}},
)

@router.get("/", response_model=List[DailyLogResponse])
async def list_daily_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Lista todos los registros diarios."""
    logs = db.query(DailyLog).offset(skip).limit(limit).all()
    return logs
