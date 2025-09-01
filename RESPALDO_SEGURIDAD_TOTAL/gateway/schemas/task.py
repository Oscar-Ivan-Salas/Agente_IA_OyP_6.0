"""
Esquemas para la gestión de tareas.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from ..pyd_compat import Field, validator, root_validator
from .base import BaseSchema, TimestampMixin, IDMixin

class TaskStatus(str, Enum):
    """Estados posibles de una tarea."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class TaskPriority(str, Enum):
    """Niveles de prioridad de tareas."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskBase(BaseSchema):
    """Esquema base para tareas."""
    project_id: str = Field(..., description="ID del proyecto al que pertenece la tarea")
    parent_id: Optional[str] = Field(None, description="ID de la tarea padre (para subtareas)")
    wbs: str = Field(..., max_length=50, description="Estructura de desglose del trabajo (WBS)")
    name: str = Field(..., max_length=255, description="Nombre de la tarea")
    description: Optional[str] = Field(None, description="Descripción detallada de la tarea")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Estado actual de la tarea")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Prioridad de la tarea")
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Progreso de la tarea (0-100)")
    estimated_hours: float = Field(0.0, ge=0.0, description="Horas estimadas para completar")
    actual_hours: float = Field(0.0, ge=0.0, description="Horas reales trabajadas")
    start_date: Optional[datetime] = Field(None, description="Fecha de inicio planificada")
    due_date: Optional[datetime] = Field(None, description="Fecha de vencimiento")
    completed_at: Optional[datetime] = Field(None, description="Fecha de finalización")

    @root_validator(pre=True, skip_on_failure=True)
    def validate_dates(cls, values):
        """Valida que la fecha de inicio sea anterior a la de vencimiento."""
        start_date = values.get('start_date')
        due_date = values.get('due_date')
        
        if start_date and due_date and start_date > due_date:
            raise ValueError("La fecha de inicio debe ser anterior a la fecha de vencimiento")
            
        return values

class TaskCreate(TaskBase):
    """Esquema para la creación de tareas."""
    pass

class TaskUpdate(BaseSchema):
    """Esquema para la actualización de tareas."""
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    estimated_hours: Optional[float] = Field(None, ge=0.0)
    actual_hours: Optional[float] = Field(None, ge=0.0)
    start_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class TaskResponse(TaskBase, IDMixin, TimestampMixin):
    """Esquema para la respuesta de tareas."""
    is_active: bool = Field(True, description="Indica si la tarea está activa")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "parent_id": None,
                "wbs": "1.1",
                "name": "Configurar servidor de base de datos",
                "description": "Instalar y configurar el servidor de base de datos",
                "status": "in_progress",
                "priority": "high",
                "progress": 50.0,
                "estimated_hours": 8.0,
                "actual_hours": 4.0,
                "start_date": "2023-01-01T09:00:00",
                "due_date": "2023-01-05T18:00:00",
                "completed_at": None,
                "is_active": True,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-03T12:00:00"
            }
        }

class TaskWithRelations(TaskResponse):
    """Esquema que incluye las relaciones de la tarea."""
    project: Optional["ProjectResponse"] = Field(None, description="Proyecto al que pertenece")
    parent: Optional["TaskResponse"] = Field(None, description="Tarea padre")
    subtasks: List["TaskResponse"] = Field(default_factory=list, description="Subtareas")

# Las referencias circulares se resuelven en schemas/__init__.py
