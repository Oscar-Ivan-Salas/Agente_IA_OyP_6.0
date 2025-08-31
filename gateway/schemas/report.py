"""
Esquemas para la gestión de reportes.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import Field, validator
from .base import BaseSchema, TimestampMixin, IDMixin

class ReportType(str, Enum):
    """Tipos de reportes disponibles."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    PROJECT_SUMMARY = "project_summary"
    RISK_ANALYSIS = "risk_analysis"
    TASK_PROGRESS = "task_progress"
    CUSTOM = "custom"

class ReportStatus(str, Enum):
    """Estados posibles de un reporte."""
    DRAFT = "draft"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"

class ReportBase(BaseSchema):
    """Esquema base para reportes."""
    project_id: str = Field(..., description="ID del proyecto relacionado")
    report_type: ReportType = Field(..., description="Tipo de reporte")
    title: str = Field(..., max_length=255, description="Título del reporte")
    description: Optional[str] = Field(None, description="Descripción del reporte")
    status: ReportStatus = Field(ReportStatus.DRAFT, description="Estado actual del reporte")
    file_path: Optional[str] = Field(None, description="Ruta al archivo generado")
    start_date: Optional[datetime] = Field(None, description="Fecha de inicio del período del reporte")
    end_date: Optional[datetime] = Field(None, description="Fecha de fin del período del reporte")
    generated_at: Optional[datetime] = Field(None, description="Fecha de generación del reporte")

class ReportCreate(ReportBase):
    """Esquema para la creación de reportes."""
    report_type: Optional[ReportType] = ReportType.CUSTOM
    status: ReportStatus = ReportStatus.DRAFT
    
    @validator('title', pre=True, always=True)
    def set_default_title(cls, v, values):
        if v is None and 'report_type' in values and 'start_date' in values:
            report_type = values['report_type']
            start_date = values.get('start_date')
            date_str = start_date.strftime('%Y-%m-%d') if start_date else ''
            return f"{report_type.replace('_', ' ').title()} Report - {date_str}"
        return v

class ReportUpdate(BaseSchema):
    """Esquema para la actualización de reportes."""
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ReportStatus] = None
    file_path: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    generated_at: Optional[datetime] = None

class ReportResponse(ReportBase, IDMixin, TimestampMixin):
    """Esquema para la respuesta de reportes."""
    is_active: bool = Field(True, description="Indica si el reporte está activo")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "880e8400-e29b-41d4-a716-446655440004",
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "report_type": "weekly",
                "title": "Weekly Progress Report - 2023-06-01",
                "description": "Reporte semanal de avance del proyecto",
                "status": "completed",
                "file_path": "/reports/weekly_20230601.pdf",
                "start_date": "2023-05-29T00:00:00",
                "end_date": "2023-06-04T23:59:59",
                "generated_at": "2023-06-04T18:30:00",
                "is_active": True,
                "created_at": "2023-06-01T09:00:00",
                "updated_at": "2023-06-04T18:35:00"
            }
        }

class ReportWithRelations(ReportResponse):
    """Esquema que incluye las relaciones del reporte."""
    project: Optional["ProjectResponse"] = Field(None, description="Proyecto relacionado")

# Resolver referencias circulares
def _resolve_forward_refs():
    from .project import ProjectResponse
    ReportWithRelations.update_forward_refs(ProjectResponse=ProjectResponse)

_resolve_forward_refs()
