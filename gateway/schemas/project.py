"""
Esquemas para la gestión de proyectos.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import Field, validator
from .base import BaseSchema, TimestampMixin, IDMixin

class ProjectBase(BaseSchema):
    """Esquema base para proyectos."""
    name: str = Field(..., max_length=255, description="Nombre del proyecto")
    description: Optional[str] = Field(None, description="Descripción detallada del proyecto")
    start_date: Optional[datetime] = Field(None, description="Fecha de inicio planificada")
    due_date: Optional[datetime] = Field(None, description="Fecha de entrega estimada")
    status: str = Field("pending", description="Estado actual del proyecto")

class ProjectCreate(ProjectBase):
    """Esquema para la creación de proyectos."""
    pass

class ProjectUpdate(BaseSchema):
    """Esquema para la actualización de proyectos."""
    name: Optional[str] = Field(None, max_length=255, description="Nombre del proyecto")
    description: Optional[str] = Field(None, description="Descripción detallada del proyecto")
    start_date: Optional[datetime] = Field(None, description="Fecha de inicio planificada")
    due_date: Optional[datetime] = Field(None, description="Fecha de entrega estimada")
    status: Optional[str] = Field(None, description="Estado actual del proyecto")

class ProjectResponse(ProjectBase, IDMixin, TimestampMixin):
    """Esquema para la respuesta de proyectos."""
    progress: float = Field(0.0, ge=0, le=100, description="Progreso general del proyecto (0-100)")
    is_active: bool = Field(True, description="Indica si el proyecto está activo")

    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Migración a la nube",
                "description": "Migración de la infraestructura a la nube pública",
                "start_date": "2023-01-01T00:00:00",
                "due_date": "2023-12-31T23:59:59",
                "status": "in_progress",
                "progress": 35.5,
                "is_active": True,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-06-01T12:00:00"
            }
        }

class ProjectWithRelations(ProjectResponse):
    """Esquema que incluye las relaciones del proyecto."""
    tasks: List["TaskResponse"] = Field(default_factory=list, description="Tareas del proyecto")
    daily_logs: List["DailyLogResponse"] = Field(default_factory=list, description="Registros diarios")
    risks: List["RiskResponse"] = Field(default_factory=list, description="Riesgos identificados")
    reports: List["ReportResponse"] = Field(default_factory=list, description="Reportes generados")

# Importaciones circulares al final
def _resolve_forward_refs():
    """Resuelve referencias circulares entre esquemas."""
    from .task import TaskResponse
    from .daily_log import DailyLogResponse
    from .risk import RiskResponse
    from .report import ReportResponse
    
    ProjectWithRelations.update_forward_refs(
        TaskResponse=TaskResponse,
        DailyLogResponse=DailyLogResponse,
        RiskResponse=RiskResponse,
        ReportResponse=ReportResponse
    )

_resolve_forward_refs()
