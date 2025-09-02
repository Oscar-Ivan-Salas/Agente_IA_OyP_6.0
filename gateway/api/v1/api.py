"""
Configuración de rutas de la API v1.
"""
from fastapi import APIRouter
from .endpoints import (
    ai,
    analytics,
    chat,
    documents,
    jobs,
    projects as projects_router,
    reports,
    tasks as tasks_router
)

# Crear el router principal de la API v1
api_router = APIRouter(prefix="/api/v1", tags=["api"])

# Incluir routers de endpoints
api_router.include_router(ai.router, prefix="/ai", tags=["ai"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(projects_router.router, prefix="/projects", tags=["projects"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(tasks_router.router, prefix="/tasks", tags=["tasks"])
