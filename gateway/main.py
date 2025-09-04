"""
Gateway principal de la aplicación.

Este módulo configura y arranca el servidor FastAPI que actúa como puerta de enlace
para todos los servicios del sistema.
"""
import json
import os
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from typing import List
import uvicorn
from pathlib import Path

from gateway.database import get_db, init_db
from gateway.routers import projects, tasks, daily_logs, risks, reports
from gateway.config import settings
from gateway.api.v1.api import api_router as v1_router

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
    allow_origins=["*"],  # Allow all origins for now, can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar templates
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Ruta para servir el frontend
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Incluir routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(daily_logs.router, prefix="/api/daily-logs", tags=["daily-logs"])
app.include_router(risks.router, prefix="/api/risks", tags=["risks"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])

# Incluir API v1
app.include_router(v1_router)

# Ruta de salud
@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud del servicio."""
    return {"status": "ok", "service": "gateway"}

# Import WebSocket manager
from .ws_app.manager import ConnectionManager

# Create WebSocket manager instance
manager = ConnectionManager()

# WebSocket Endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time communication.
    
    Args:
        websocket: The WebSocket connection
        client_id: Unique client identifier
    """
    # Accept connection and register client
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            # Parse message (expecting JSON)
            try:
                message = json.loads(data)
                message_type = message.get('type', '')
                
                # Handle different message types
                if message_type == 'subscribe':
                    # Subscribe to channels
                    channels = message.get('channels', [])
                    await manager.subscribe(client_id, channels)
                    
                elif message_type == 'unsubscribe':
                    # Unsubscribe from channels
                    channels = message.get('channels', [])
                    await manager.unsubscribe(client_id, channels)
                    
                else:
                    # Echo message back to the client (for testing)
                    await manager.send_personal_message(
                        {"type": "echo", "data": message},
                        client_id
                    )
                    
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON format"},
                    client_id
                )
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        pass

# Punto de entrada principal
if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        ws="none"  # Deshabilitar WebSockets
    )
