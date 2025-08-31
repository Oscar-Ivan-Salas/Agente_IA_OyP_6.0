"""
Esquemas para la gestión de riesgos.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import Field, validator, root_validator
from .base import BaseSchema, TimestampMixin, IDMixin

class RiskCategory(str, Enum):
    """Categorías de riesgos."""
    TECHNICAL = "technical"
    SCHEDULE = "schedule"
    RESOURCE = "resource"
    EXTERNAL = "external"
    REQUIREMENTS = "requirements"
    QUALITY = "quality"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    OTHER = "other"

class RiskStatus(str, Enum):
    """Estados posibles de un riesgo."""
    IDENTIFIED = "identified"
    MONITORED = "monitored"
    MITIGATED = "mitigated"
    OCCURRED = "occurred"
    CLOSED = "closed"

class RiskSeverity(str, Enum):
    """Niveles de severidad de riesgos."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskBase(BaseSchema):
    """Esquema base para riesgos."""
    project_id: str = Field(..., description="ID del proyecto relacionado")
    title: str = Field(..., max_length=255, description="Título del riesgo")
    description: Optional[str] = Field(None, description="Descripción detallada del riesgo")
    category: RiskCategory = Field(..., description="Categoría del riesgo")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de ocurrencia (0-1)")
    impact: float = Field(..., ge=0.0, le=1.0, description="Impacto potencial (0-1)")
    severity: RiskSeverity = Field(..., description="Severidad calculada")
    status: RiskStatus = Field(RiskStatus.IDENTIFIED, description="Estado actual del riesgo")
    mitigation_plan: Optional[str] = Field(None, description="Plan de mitigación")
    contingency_plan: Optional[str] = Field(None, description="Plan de contingencia")
    due_date: Optional[datetime] = Field(None, description="Fecha límite para la mitigación")
    resolved_at: Optional[datetime] = Field(None, description="Fecha de resolución")

    @validator('severity', pre=True, always=True)
    def calculate_severity(cls, v, values):
        """Calcula la severidad basada en probabilidad e impacto."""
        if v is not None:
            return v
            
        probability = values.get('probability', 0)
        impact = values.get('impact', 0)
        risk_score = probability * impact
        
        if risk_score >= 0.7:
            return RiskSeverity.CRITICAL
        elif risk_score >= 0.5:
            return RiskSeverity.HIGH
        elif risk_score >= 0.3:
            return RiskSeverity.MEDIUM
        else:
            return RiskSeverity.LOW

class RiskCreate(RiskBase):
    """Esquema para la creación de riesgos."""
    severity: Optional[RiskSeverity] = None  # Se calcula automáticamente si no se proporciona

class RiskUpdate(BaseSchema):
    """Esquema para la actualización de riesgos."""
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[RiskCategory] = None
    probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    impact: Optional[float] = Field(None, ge=0.0, le=1.0)
    severity: Optional[RiskSeverity] = None
    status: Optional[RiskStatus] = None
    mitigation_plan: Optional[str] = None
    contingency_plan: Optional[str] = None
    due_date: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

class RiskResponse(RiskBase, IDMixin, TimestampMixin):
    """Esquema para la respuesta de riesgos."""
    is_active: bool = Field(True, description="Indica si el riesgo está activo")
    risk_score: float = Field(..., description="Puntuación de riesgo (probabilidad * impacto)")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "770e8400-e29b-41d4-a716-446655440003",
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Falta de personal calificado",
                "description": "Posible escasez de desarrolladores con experiencia en la tecnología requerida.",
                "category": "resource",
                "probability": 0.7,
                "impact": 0.8,
                "severity": "high",
                "risk_score": 0.56,
                "status": "monitored",
                "mitigation_plan": "Contratar un consultor externo temporal.",
                "contingency_plan": "Redistribuir tareas entre el equipo actual.",
                "due_date": "2023-06-30T00:00:00",
                "resolved_at": None,
                "is_active": True,
                "created_at": "2023-05-15T10:30:00",
                "updated_at": "2023-05-20T14:45:00"
            }
        }

class RiskWithRelations(RiskResponse):
    """Esquema que incluye las relaciones del riesgo."""
    project: Optional["ProjectResponse"] = Field(None, description="Proyecto relacionado")

# Resolver referencias circulares
def _resolve_forward_refs():
    from .project import ProjectResponse
    RiskWithRelations.update_forward_refs(ProjectResponse=ProjectResponse)

_resolve_forward_refs()
