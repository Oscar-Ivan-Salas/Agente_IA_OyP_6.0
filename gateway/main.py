"""
Gateway principal de la aplicación.

Este módulo configura y arranca el servidor FastAPI que actúa como puerta de enlace
para todos los servicios del sistema.
"""
import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import List
import uvicorn

from .database import get_db, init_db
from .routers import projects, tasks, daily_logs, risks, reports
from .config import settings

# Configuración del ciclo de vida de la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta al iniciar la aplicación
    print("🚀 Iniciando Gateway...")
    
    # Inicializar la base de datos
    init_db()
    
    # Código que se ejecuta al apagar la aplicación
    yield
    print("👋 Deteniendo Gateway...")

# Crear la aplicación FastAPI
app = FastAPI(
    title="Agente IA OYP - Gateway",
    description="API Gateway para el sistema de gestión de proyectos OYP",
    version="0.1.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Incluir routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(daily_logs.router, prefix="/api/daily-logs", tags=["daily-logs"])
app.include_router(risks.router, prefix="/api/risks", tags=["risks"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])

# Ruta de salud
@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud del servicio."""
    return {"status": "ok", "service": "gateway"}

# Punto de entrada principal
if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )
