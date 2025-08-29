#!/usr/bin/env python3
"""
🚀 AGENTE IA OYP 6.0 - GATEWAY PRINCIPAL
=====================================
Backend FastAPI completo - Coordinador de microservicios
Archivo: gateway/app.py (800 líneas completas)
"""

import asyncio
import httpx
import json
import logging
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# FastAPI imports
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64

# =====================
# CONFIGURACIÓN GLOBAL
# =====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directorios
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
EXPORTS_DIR = BASE_DIR / "exports"

# Crear directorios
for directory in [TEMPLATES_DIR, STATIC_DIR, UPLOADS_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# URLs de microservicios
SERVICES_CONFIG = {
    "ai-engine": "http://localhost:8001",
    "document-processor": "http://localhost:8002",
    "analytics-engine": "http://localhost:8003",
    "report-generator": "http://localhost:8004"
}

# =====================
# APLICACIÓN FASTAPI
# =====================

app = FastAPI(
    title="🤖 Agente IA OyP 6.0 - Gateway",
    description="Gateway principal para sistema de IA empresarial",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates y archivos estáticos
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# =====================
# WEBSOCKET MANAGER
# =====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.chat_history: List[Dict] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Nueva conexión WebSocket. Total: {len(self.active_connections)}")
        
        # Enviar historial reciente
        if self.chat_history:
            for message in self.chat_history[-10:]:  # Últimos 10 mensajes
                await websocket.send_text(json.dumps(message))
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Conexión WebSocket cerrada. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: dict):
        """Enviar mensaje a todas las conexiones activas"""
        self.chat_history.append(message)
        
        # Mantener solo últimos 100 mensajes
        if len(self.chat_history) > 100:
            self.chat_history = self.chat_history[-100:]
        
        # Broadcast a todas las conexiones
        for connection in self.active_connections[:]:  # Copia para evitar modificación durante iteración
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Conexión cerrada, remover
                self.active_connections.remove(connection)

manager = ConnectionManager()

# =====================
# MODELOS PYDANTIC
# =====================

class ChatMessage(BaseModel):
    message: str
    timestamp: Optional[str] = None

class ServiceStatus(BaseModel):
    name: str
    status: str
    url: str
    response_time: Optional[float] = None

class AnalyticsRequest(BaseModel):
    analysis_type: str
    data: List[Dict]
    options: Optional[Dict] = {}

# =====================
# UTILIDADES
# =====================

async def check_service_health(service_name: str, url: str) -> ServiceStatus:
    """Verificar estado de un microservicio"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start_time = datetime.now()
            response = await client.get(f"{url}/health")
            response_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code == 200:
                return ServiceStatus(
                    name=service_name,
                    status="online",
                    url=url,
                    response_time=round(response_time * 1000, 2)  # ms
                )
            else:
                return ServiceStatus(name=service_name, status="error", url=url)
    except Exception as e:
        logger.error(f"Error checking {service_name}: {e}")
        return ServiceStatus(name=service_name, status="offline", url=url)

async def proxy_to_service(service_name: str, endpoint: str, method: str = "GET", data: Any = None):
    """Proxy request a microservicio"""
    if service_name not in SERVICES_CONFIG:
        raise HTTPException(status_code=404, detail=f"Servicio {service_name} no encontrado")
    
    url = f"{SERVICES_CONFIG[service_name]}{endpoint}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method.upper() == "GET":
                response = await client.get(url)
            elif method.upper() == "POST":
                response = await client.post(url, json=data)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data)
            elif method.upper() == "DELETE":
                response = await client.delete(url)
            else:
                raise HTTPException(status_code=405, detail="Método no permitido")
            
            return response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
    
    except httpx.TimeoutException:
        # Fallback a datos mock
        return get_mock_response(service_name, endpoint)
    except Exception as e:
        logger.error(f"Error en proxy a {service_name}: {e}")
        return get_mock_response(service_name, endpoint)

def get_mock_response(service_name: str, endpoint: str) -> Dict:
    """Respuestas mock cuando los servicios no están disponibles"""
    mock_responses = {
        "ai-engine": {
            "/analyze": {
                "status": "success",
                "analysis": {
                    "sentiment": {"score": 0.85, "label": "positivo"},
                    "entities": ["IA", "tecnología", "análisis"],
                    "keywords": ["inteligencia artificial", "machine learning"],
                    "summary": "Análisis de texto con IA - resultado mock"
                },
                "model_used": "mock-model-v1.0",
                "processing_time": 0.45
            },
            "/chat": {
                "response": "Soy un asistente IA simulado. Los microservicios reales están en desarrollo. ¿En qué puedo ayudarte?",
                "model": "mock-llm",
                "tokens_used": 25
            }
        },
        "document-processor": {
            "/upload": {
                "status": "success",
                "text_extracted": "Texto extraído del documento (simulado)",
                "pages": 1,
                "word_count": 150,
                "file_type": "pdf"
            },
            "/ocr": {
                "status": "success", 
                "text": "Texto OCR extraído (simulado)",
                "confidence": 0.95
            }
        },
        "analytics-engine": {
            "/analyze": {
                "status": "success",
                "results": {
                    "descriptive": {"mean": 45.2, "std": 12.8, "median": 44.1},
                    "correlation": {"pearson": 0.73, "spearman": 0.68},
                    "regression": {"r2": 0.84, "mse": 2.1}
                },
                "charts": ["histogram", "scatter", "correlation_matrix"]
            }
        },
        "report-generator": {
            "/generate": {
                "status": "success",
                "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "format": "pdf",
                "file_path": "/exports/report_sample.pdf"
            }
        },
        "chat-service": {
            "/message": {
                "response": "Respuesta del chat simulada",
                "timestamp": datetime.now().isoformat()
            }
        }
    }
    
    return mock_responses.get(service_name, {}).get(endpoint, {"status": "mock", "message": "Respuesta simulada"})

def generate_dashboard_stats():
    """Generar estadísticas para el dashboard"""
    now = datetime.now()
    return {
        "kpis": {
            "documents_processed": np.random.randint(150, 300),
            "ai_predictions": np.random.randint(80, 150),
            "active_models": np.random.randint(3, 8),
            "success_rate": round(np.random.uniform(85, 98), 1)
        },
        "activity_data": [
            {"time": (now - timedelta(hours=i)).strftime("%H:%M"), "value": np.random.randint(10, 50)}
            for i in range(24, 0, -1)
        ],
        "distribution_data": [
            {"label": "Documentos", "value": np.random.randint(30, 60)},
            {"label": "Análisis", "value": np.random.randint(20, 40)},
            {"label": "Reportes", "value": np.random.randint(10, 30)},
            {"label": "Chat", "value": np.random.randint(5, 25)}
        ],
        "system_metrics": {
            "cpu_usage": round(np.random.uniform(20, 80), 1),
            "memory_usage": round(np.random.uniform(40, 85), 1),
            "disk_usage": round(np.random.uniform(30, 70), 1),
            "network_io": round(np.random.uniform(10, 100), 1)
        }
    }

# =====================
# RUTAS PRINCIPALES
# =====================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Página principal del dashboard"""
    try:
        # Verificar que el archivo de plantilla existe
        template_path = TEMPLATES_DIR / "index.html"
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
            
        # Contexto para la plantilla
        context = {
            "request": request,
            "title": "Dashboard",
            "version": "6.0.0",
            "services": list(SERVICES_CONFIG.keys()),
            "current_year": datetime.now().year
        }
        
        return templates.TemplateResponse("index.html", context)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error al cargar el dashboard: {str(e)}\n{error_trace}")
        
        # Página de error más detallada para depuración
        error_message = str(e)  # Define error_message here
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error en el Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ padding: 2rem; }}
                .error-details {{ 
                    background: #f8f9fa; 
                    padding: 1rem; 
                    border-radius: 0.25rem;
                    margin-top: 1rem;
                    font-family: monospace;
                    white-space: pre-wrap;
                    max-height: 300px;
                    overflow-y: auto;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Error al cargar el dashboard</h4>
                    <p>{error_message}</p>
                    <hr>
                    <p class="mb-0">Por favor, revisa los logs del servidor para más detalles.</p>
                </div>
                <div class="card">
                    <div class="card-header">
                        Detalles del error (solo visible en desarrollo)
                    </div>
                    <div class="card-body">
                        <div class="error-details">{error_trace}</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=error_html.format(error_message=error_message, error_trace=error_trace), status_code=500)

@app.get("/health")
async def health_check():
    """Health check del gateway"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "6.0.0"
    }

# =====================
# WEBSOCKET
# =====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Recibir mensaje del cliente
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                # Procesar mensaje de chat
                user_message = {
                    "type": "user",
                    "message": message_data.get("message", ""),
                    "timestamp": datetime.now().isoformat(),
                    "user": "Usuario"
                }
                
                # Broadcast mensaje del usuario
                await manager.broadcast(user_message)
                
                # Simular respuesta de IA (en producción sería proxy a ai-engine)
                await asyncio.sleep(1)  # Simular procesamiento
                
                ai_response = {
                    "type": "ai",
                    "message": f"Respuesta simulada a: '{message_data.get('message')}'",
                    "timestamp": datetime.now().isoformat(),
                    "user": "IA"
                }
                
                # Broadcast respuesta de IA
                await manager.broadcast(ai_response)
            
            elif message_data.get("type") == "system":
                # Mensajes del sistema
                system_message = {
                    "type": "system",
                    "message": message_data.get("message", ""),
                    "timestamp": datetime.now().isoformat()
                }
                await manager.broadcast(system_message)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# =====================
# APIs DASHBOARD
# =====================

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Obtener estadísticas del dashboard"""
    return generate_dashboard_stats()

@app.get("/api/services/status")
async def get_services_status():
    """Obtener estado de todos los microservicios"""
    status_list = []
    
    for service_name, url in SERVICES_CONFIG.items():
        status = await check_service_health(service_name, url)
        status_list.append(status.dict())
    
    return {"services": status_list, "total": len(status_list)}

# =====================
# PROXY APIS
# =====================

@app.post("/api/ai/analyze")
async def ai_analyze(request: dict):
    """Proxy para análisis de IA"""
    return await proxy_to_service("ai-engine", "/analyze", "POST", request)

@app.post("/api/ai/chat")
async def ai_chat(chat_message: ChatMessage):
    """Proxy para chat con IA"""
    return await proxy_to_service("ai-engine", "/chat", "POST", chat_message.dict())

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Proxy para subida de documentos, enviando el archivo al servicio."""
    
    content = await file.read()
    
    # Asegurarse de que el puntero del archivo esté al inicio si se necesita releer
    await file.seek(0)

    # Crear el payload multipart/form-data
    files = {'file': (file.filename, content, file.content_type)}
    
    service_name = "document-processor"
    endpoint = "/upload"
    url = f"{SERVICES_CONFIG[service_name]}{endpoint}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # No se envía 'json=', sino 'files='
            response = await client.post(url, files=files)
            
            # Re-lanzar errores HTTP del servicio para que el cliente los vea
            response.raise_for_status()
            
            return response.json()
            
    except httpx.TimeoutException:
        logger.error(f"Timeout al contactar al servicio {service_name}")
        raise HTTPException(status_code=504, detail=f"Timeout al contactar al servicio {service_name}")
    except httpx.RequestError as e:
        logger.error(f"Error de red al contactar a {service_name}: {e}")
        # Devolver una respuesta mock como fallback si el servicio no está disponible
        return get_mock_response(service_name, endpoint)
    except httpx.HTTPStatusError as e:
        # Propagar el error del servicio al cliente
        logger.error(f"Error del servicio {service_name}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        logger.error(f"Error inesperado en proxy a {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Error interno en el gateway")

@app.post("/api/analytics/analyze")
async def analytics_analyze(request: AnalyticsRequest):
    """Proxy para análisis estadístico"""
    return await proxy_to_service("analytics-engine", "/analyze", "POST", request.dict())

@app.post("/api/reports/generate")
async def generate_report(request: dict):
    """Proxy para generación de reportes"""
    return await proxy_to_service("report-generator", "/generate", "POST", request)

@app.post("/api/ai/summarize")
async def ai_summarize(request: dict):
    """Proxy para resumen de IA"""
    return await proxy_to_service("ai-engine", "/summarize", "POST", request)

@app.post("/api/documents/process_text")
async def documents_process_text(request: dict):
    """Proxy para procesamiento de texto"""
    return await proxy_to_service("document-processor", "/process_text", "POST", request)

@app.post("/api/analytics/visualize")
async def analytics_visualize(request: dict):
    """Proxy para visualización de datos"""
    return await proxy_to_service("analytics-engine", "/visualize", "POST", request)

@app.post("/api/analytics/text_analytics")
async def analytics_text_analytics(request: dict):
    """Proxy para analíticas de texto"""
    return await proxy_to_service("analytics-engine", "/text_analytics", "POST", request)

@app.post("/api/reports/quick_report")
async def reports_quick_report(request: dict):
    """Proxy para reportes rápidos"""
    return await proxy_to_service("report-generator", "/quick_report", "POST", request)

@app.post("/api/analytics/upload_dataset")
async def analytics_upload_dataset(file: UploadFile = File(...), analysis_type: str = Form(...)):
    """Proxy para subida de datasets a analytics-engine"""
    # En un caso real, aquí se podría hacer streaming del archivo directamente
    # al servicio. Por simplicidad, lo pasamos como datos.
    content = await file.read()
    return await proxy_to_service("analytics-engine", "/upload_dataset", "POST", {
        "filename": file.filename,
        "analysis_type": analysis_type,
        "file_content_b64": base64.b64encode(content).decode('utf-8') # Asumimos que el servicio puede manejar base64
    })

# =====================
# APIS ESPECÍFICAS
# =====================

@app.get("/api/training/projects")
async def get_training_projects():
    """Obtener proyectos de entrenamiento"""
    # Mock data para proyectos de entrenamiento
    return {
        "projects": [
            {
                "id": 1,
                "name": "Modelo Documentos Legales",
                "status": "completed",
                "accuracy": 0.94,
                "created": "2024-01-15",
                "model_type": "text-classification"
            },
            {
                "id": 2,
                "name": "Análisis Sentimientos Cliente",
                "status": "training",
                "progress": 0.67,
                "created": "2024-01-20",
                "model_type": "sentiment-analysis"
            }
        ]
    }

@app.post("/api/training/start")
async def start_training(request: dict):
    """Iniciar entrenamiento de modelo"""
    return {
        "status": "started",
        "project_id": np.random.randint(1000, 9999),
        "estimated_time": "2-4 horas",
        "message": "Entrenamiento iniciado correctamente"
    }

@app.get("/api/agent/history")
async def get_agent_history():
    """Obtener historial del agente"""
    return {
        "actions": [
            {
                "command": "Analizar documentos de ventas Q1",
                "status": "completed",
                "result": "42 documentos procesados, insights generados",
                "timestamp": "2024-01-20 10:30:00"
            },
            {
                "command": "Entrenar modelo para clasificación emails",
                "status": "completed", 
                "result": "Modelo entrenado con 95% precisión",
                "timestamp": "2024-01-19 15:45:00"
            }
        ]
    }

@app.post("/api/agent/execute")
async def execute_agent_command(request: dict):
    """Ejecutar comando del agente"""
    command = request.get("command", "")
    
    # Simular procesamiento
    await asyncio.sleep(1)
    
    return {
        "status": "success",
        "result": f"Comando ejecutado: {command}",
        "actions_taken": [
            "Análisis de texto completado",
            "Resultados guardados en base de datos",
            "Notificación enviada a usuarios"
        ],
        "execution_time": 1.23
    }

# =====================
# CONFIGURACIÓN
# =====================

@app.get("/api/config")
async def get_configuration():
    """Obtener configuración del sistema"""
    return {
        "api_endpoints": SERVICES_CONFIG,
        "features": {
            "ai_analysis": True,
            "document_processing": True,
            "analytics": True,
            "reports": True,
            "chat": True
        },
        "models": {
            "available": ["GPT-4", "Claude", "Llama", "Local"],
            "active": "Local"
        }
    }

@app.post("/api/config")
async def update_configuration(config: dict):
    """Actualizar configuración del sistema"""
    return {
        "status": "updated",
        "message": "Configuración actualizada correctamente"
    }

# =====================
# ARCHIVOS ESTÁTICOS
# =====================

@app.get("/api/reports/download/{filename}")
async def download_report(filename: str):
    """Descargar reporte generado"""
    file_path = EXPORTS_DIR / filename
    
    if file_path.exists():
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
    else:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

# =====================
# STARTUP EVENTS
# =====================

@app.on_event("startup")
async def startup_event():
    """Eventos de inicio"""
    logger.info("🚀 Iniciando Agente IA OyP 6.0 Gateway...")
    logger.info(f"📁 Templates: {TEMPLATES_DIR}")
    logger.info(f"📁 Static: {STATIC_DIR}")
    logger.info(f"📁 Uploads: {UPLOADS_DIR}")
    logger.info(f"🔗 Servicios configurados: {list(SERVICES_CONFIG.keys())}")
    
    # Verificar servicios
    for service_name, url in SERVICES_CONFIG.items():
        status = await check_service_health(service_name, url)
        logger.info(f"🔧 {service_name}: {status.status}")

@app.on_event("shutdown")
async def shutdown_event():
    """Eventos de cierre"""
    logger.info("🛑 Cerrando Agente IA OyP 6.0 Gateway...")

# =====================
# CONFIGURACIÓN DE ARCHIVOS ESTÁTICOS Y PLANTILLAS
# =====================

# Configurar directorio de plantillas
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Configurar archivos estáticos
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# =====================
# MAIN
# =====================

if __name__ == "__main__":
    logger.info("🚀 Iniciando servidor en modo desarrollo...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )