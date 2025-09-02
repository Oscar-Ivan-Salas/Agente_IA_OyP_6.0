"""
Endpoints para la gestión de proyectos.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from ..database import get_db
from ..models import Project, Task, DailyLog, Risk, Report
from ..schemas import (
    ProjectCreate, ProjectUpdate, ProjectResponse,
    TaskResponse, DailyLogResponse, RiskResponse, ReportResponse
)

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """
    Crea un nuevo proyecto.
    """
    db_project = Project(**project.dict())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@router.get("/", response_model=List[ProjectResponse])
async def list_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Lista todos los proyectos.
    """
    return db.query(Project).filter(Project.is_active).offset(skip).limit(limit).all()

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: UUID, db: Session = Depends(get_db)):
    """
    Obtiene un proyecto por su ID.
    """
    project = db.query(Project).filter(Project.id == project_id, Project.is_active).first()
    if not project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    return project

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID, 
    project_update: ProjectUpdate, 
    db: Session = Depends(get_db)
):
    """
    Actualiza un proyecto existente.
    """
    db_project = db.query(Project).filter(Project.id == project_id, Project.is_active).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    
    update_data = project_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_project, field, value)
    
    db.commit()
    db.refresh(db_project)
    return db_project

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: UUID, db: Session = Depends(get_db)):
    """
    Elimina un proyecto (borrado lógico).
    """
    db_project = db.query(Project).filter(Project.id == project_id, Project.is_active).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    
    db_project.is_active = False
    db.commit()
    return None

# Endpoints relacionados con tareas
@router.get("/{project_id}/tasks", response_model=List[TaskResponse])
async def list_project_tasks(project_id: UUID, db: Session = Depends(get_db)):
    """
    Lista todas las tareas de un proyecto.
    """
    project = db.query(Project).filter(Project.id == project_id, Project.is_active).first()
    if not project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    
    return db.query(Task).filter(Task.project_id == project_id, Task.is_active).all()

# Endpoints relacionados con registros diarios
@router.get("/{project_id}/daily-logs", response_model=List[DailyLogResponse])
async def list_project_daily_logs(project_id: UUID, db: Session = Depends(get_db)):
    """
    Lista todos los registros diarios de un proyecto.
    """
    project = db.query(Project).filter(Project.id == project_id, Project.is_active).first()
    if not project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    
    return db.query(DailyLog).filter(DailyLog.project_id == project_id).order_by(DailyLog.log_date.desc()).all()

# Endpoints relacionados con riesgos
@router.get("/{project_id}/risks", response_model=List[RiskResponse])
async def list_project_risks(project_id: UUID, db: Session = Depends(get_db)):
    """
    Lista todos los riesgos de un proyecto.
    """
    project = db.query(Project).filter(Project.id == project_id, Project.is_active).first()
    if not project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    
    return db.query(Risk).filter(Risk.project_id == project_id, Risk.is_active).all()

# Endpoints relacionados con reportes
@router.get("/{project_id}/reports", response_model=List[ReportResponse])
async def list_project_reports(project_id: UUID, db: Session = Depends(get_db)):
    """
    Lista todos los reportes de un proyecto.
    """
    project = db.query(Project).filter(Project.id == project_id, Project.is_active).first()
    if not project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    
    return db.query(Report).filter(Report.project_id == project_id, Report.is_active).order_by(Report.created_at.desc()).all()
