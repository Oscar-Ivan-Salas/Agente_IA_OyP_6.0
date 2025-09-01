"""
Esquemas base para la API.

Contiene las clases base que son heredadas por otros esquemas.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from ..pyd_compat import BaseModel, Field
from uuid import UUID

class BaseSchema(BaseModel):
    """Esquema base con configuración común."""
    class Config:
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: str
        }
        
        # Configuración para manejo de tipos opcionales
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: str
        }

class TimestampMixin(BaseModel):
    """Campos de marca de tiempo comunes."""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class IDMixin(BaseModel):
    """Campo ID común."""
    id: UUID

class PaginationParams(BaseModel):
    """Parámetros de paginación."""
    skip: int = 0
    limit: int = 100

class PaginatedResponse(BaseModel):
    """Respuesta paginada genérica."""
    total: int
    items: List[Any]
    skip: int
    limit: int

class ErrorResponse(BaseModel):
    """Respuesta de error estándar."""
    detail: str
    code: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class SuccessResponse(BaseModel):
    """Respuesta de éxito estándar."""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None

def create_response_schema(schema: type[BaseModel], name: str = None):
    """
    Crea un esquema de respuesta con metadatos.
    
    Args:
        schema: El esquema de datos a envolver.
        name: Nombre opcional para el modelo generado.
    """
    class ResponseModel(BaseModel):
        success: bool = True
        data: schema
        meta: Optional[Dict[str, Any]] = None
        
        class Config:
            orm_mode = True
            json_encoders = {
                datetime: lambda v: v.isoformat(),
                UUID: str
            }
    
    if name:
        ResponseModel.__name__ = name
    
    return ResponseModel
