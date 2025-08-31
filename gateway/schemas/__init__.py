"""
Esquemas Pydantic para validación de datos.

Este módulo contiene todos los esquemas utilizados para validar los datos
de entrada y salida de la API.
"""
# Importaciones base primero
from .base import *

# Importaciones de esquemas individuales
from .project import (
    ProjectBase, ProjectCreate, ProjectUpdate, 
    ProjectResponse, ProjectWithRelations
)

from .task import (
    TaskStatus, TaskPriority, TaskBase, 
    TaskCreate, TaskUpdate, TaskResponse, TaskWithRelations
)

from .daily_log import (
    DailyLogBase, DailyLogCreate, DailyLogUpdate, 
    DailyLogResponse, DailyLogWithRelations
)

from .risk import (
    RiskSeverity, RiskStatus, 
    RiskBase, RiskCreate, RiskUpdate, RiskResponse, RiskWithRelations
)

from .report import (
    ReportType, ReportStatus, ReportBase, 
    ReportCreate, ReportUpdate, ReportResponse, ReportWithRelations
)

# Resolver referencias circulares
ProjectResponse.update_forward_refs(
    TaskResponse=TaskResponse,
    DailyLogResponse=DailyLogResponse,
    RiskResponse=RiskResponse,
    ReportResponse=ReportResponse
)

TaskResponse.update_forward_refs(
    ProjectResponse=ProjectResponse,
    TaskResponse=TaskResponse
)

# Nota: Agrega actualizaciones similares para otros modelos con referencias circulares si es necesario
