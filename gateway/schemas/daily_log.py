"""
Esquemas para la gestión de registros diarios.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from enum import Enum
from pydantic import Field, validator, BaseModel
from .base import BaseSchema, TimestampMixin, IDMixin

# Para evitar importaciones circulares
if TYPE_CHECKING:
    from .project import ProjectResponse

class MoodLevel(str, Enum):
    """Niveles de ánimo para los registros diarios."""
    VERY_GOOD = "very_good"
    GOOD = "good"
    NEUTRAL = "neutral"
    BAD = "bad"
    VERY_BAD = "very_bad"

class DailyLogBase(BaseSchema):
    """Esquema base para registros diarios."""
    project_id: str = Field(..., description="ID del proyecto relacionado")
    log_date: datetime = Field(..., description="Fecha del registro")
    summary: str = Field(..., description="Resumen del día")
    details: Optional[str] = Field(None, description="Detalles adicionales")
    hours_worked: float = Field(0.0, ge=0.0, description="Horas trabajadas")
    blockers: Optional[str] = Field(None, description="Bloqueadores o impedimentos")
    next_steps: Optional[str] = Field(None, description="Próximos pasos")
    mood: Optional[MoodLevel] = Field(None, description="Estado de ánimo")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Datos adicionales")

class DailyLogCreate(DailyLogBase):
    """Esquema para la creación de registros diarios."""
    log_date: Optional[datetime] = Field(None, description="Fecha del registro (opcional, por defecto ahora)")
    
    @validator('log_date', pre=True, always=True)
    def set_default_log_date(cls, v):
        return v or datetime.utcnow()

class DailyLogUpdate(BaseSchema):
    """Esquema para la actualización de registros diarios."""
    summary: Optional[str] = None
    details: Optional[str] = None
    hours_worked: Optional[float] = Field(None, ge=0.0)
    blockers: Optional[str] = None
    next_steps: Optional[str] = None
    mood: Optional[MoodLevel] = None
    additional_data: Optional[Dict[str, Any]] = None

class DailyLogResponse(DailyLogBase, IDMixin, TimestampMixin):
    """Esquema para la respuesta de registros diarios."""
    is_active: bool = Field(True, description="Indica si el registro está activo")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "660e8400-e29b-41d4-a716-446655440002",
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "log_date": "2023-06-01T18:00:00",
                "summary": "Avance en la configuración del servidor",
                "details": "Se completó la instalación del servidor de base de datos y se configuraron los usuarios básicos.",
                "hours_worked": 8.0,
                "blockers": "Falta de acceso a la documentación del servidor",
                "next_steps": "Configurar las réplicas y el respaldo automático",
                "mood": "good",
                "additional_data": {"meeting_notes": "Reunión con el equipo a las 10:00"},
                "is_active": True,
                "created_at": "2023-06-01T19:30:00",
                "updated_at": "2023-06-01T19:30:00"
            }
        }

class DailyLogWithRelations(DailyLogResponse):
    """Esquema que incluye las relaciones del registro diario."""
    project: Optional["ProjectResponse"] = Field(None, description="Proyecto relacionado")

# Función para resolver referencias circulares
def _resolve_daily_log_refs():
    """Resuelve referencias circulares para el esquema de registros diarios."""
    from .project import ProjectResponse
    
    # Actualizar referencias circulares
    DailyLogWithRelations.update_forward_refs(
        ProjectResponse=ProjectResponse
    )

_resolve_daily_log_refs()
