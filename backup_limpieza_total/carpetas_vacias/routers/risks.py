"""
Endpoints para la gestión de riesgos.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from ..database import get_db
from ..models import Risk, Project
from ..schemas import RiskCreate, RiskUpdate, RiskResponse, ProjectResponse

router = APIRouter(
    prefix="/api/risks",
    tags=["risks"],
    responses={404: {"description": "No encontrado"}},
)

@router.get("/", response_model=List[RiskResponse])
async def list_risks(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Lista todos los riesgos."""
    risks = db.query(Risk).offset(skip).limit(limit).all()
    return risks
