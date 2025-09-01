"""
Modelos para el sistema de orquestación de trabajos.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID, uuid4
from ..pyd_compat import BaseModel, Field

class JobStatus(str, Enum):
    """Estados posibles de un trabajo."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobType(str, Enum):
    """Tipos de trabajos soportados."""
    DOCUMENT_PROCESSING = "document_processing"
    AI_ANALYSIS = "ai_analysis"
    REPORT_GENERATION = "report_generation"
    DATA_SYNC = "data_sync"

class Job(BaseModel):
    """Modelo que representa un trabajo en el sistema."""
    id: UUID = Field(default_factory=uuid4, description="Identificador único del trabajo")
    type: JobType = Field(..., description="Tipo de trabajo")
    status: JobStatus = Field(default=JobStatus.PENDING, description="Estado actual del trabajo")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progreso del trabajo (0-100)")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Resultado del trabajo")
    error: Optional[str] = Field(default=None, description="Mensaje de error si el trabajo falla")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Fecha de creación")
    started_at: Optional[datetime] = Field(default=None, description="Fecha de inicio de ejecución")
    completed_at: Optional[datetime] = Field(default=None, description="Fecha de finalización")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos adicionales")

    def start(self) -> None:
        """Marca el trabajo como iniciado."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.progress = 0.0

    def update_progress(self, progress: float) -> None:
        """Actualiza el progreso del trabajo."""
        self.progress = max(0.0, min(100.0, progress))

    def complete(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Marca el trabajo como completado con éxito."""
        self.status = JobStatus.COMPLETED
        self.progress = 100.0
        self.result = result
        self.completed_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Marca el trabajo como fallido."""
        self.status = JobStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()

    def cancel(self) -> None:
        """Cancela la ejecución del trabajo."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()

class JobCreate(BaseModel):
    """Esquema para la creación de un nuevo trabajo."""
    type: JobType
    metadata: Dict[str, Any] = {}

class JobUpdate(BaseModel):
    """Esquema para actualizar un trabajo existente."""
    status: Optional[JobStatus] = None
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
