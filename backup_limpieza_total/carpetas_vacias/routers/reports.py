"""
Endpoints para la generación de reportes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from ..database import get_db
from ..models import Report, Project
from ..schemas import ReportCreate, ReportUpdate, ReportResponse, ProjectResponse

router = APIRouter(
    prefix="/api/reports",
    tags=["reports"],
    responses={404: {"description": "No encontrado"}},
)

@router.get("/", response_model=List[ReportResponse])
async def list_reports(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Lista todos los reportes."""
    reports = db.query(Report).offset(skip).limit(limit).all()
    return reports
