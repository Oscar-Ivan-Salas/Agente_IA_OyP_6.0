"""
Endpoints para la gestión de tareas.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from ..database import get_db
from ..models import Task, Project
from ..schemas import TaskCreate, TaskUpdate, TaskResponse, ProjectResponse

router = APIRouter(
    prefix="/api/tasks",
    tags=["tasks"],
    responses={404: {"description": "No encontrado"}},
)

@router.get("/", response_model=List[TaskResponse])
async def list_tasks(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Lista todas las tareas."""
    tasks = db.query(Task).offset(skip).limit(limit).all()
    return tasks
