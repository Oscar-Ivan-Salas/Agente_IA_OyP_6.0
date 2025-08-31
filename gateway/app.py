#!/usr/bin/env python3
"""
🚀 AGENTE IA OYP 6.0 - GATEWAY PRINCIPAL
=====================================
Backend FastAPI completo - Coordinador de microservicios
Archivo: gateway/app.py
"""

import asyncio
import httpx
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

# FastAPI imports
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
import uvicorn

# Importar routers
from .routes import dashboard as dashboard_routes

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

# Crear directorios
for directory in [TEMPLATES_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# --- Configuración de Microservicios ---
SERVICES_CONFIG = {
    "ai_engine": "http://localhost:8001",
    "document_processor": "http://localhost:8002",
    "analytics_engine": "http://localhost:8003",
    "report_generator": "http://localhost:8004"
}

SERVICE_TIMEOUTS = {
    "ai_engine": 30.0,
    "document_processor": 120.0,
    "analytics_engine": 90.0,
    "report_generator": 90.0,
    "default": 20.0
}

# --- Caché para el estado de los servicios ---
status_cache = {
    "timestamp": 0,
    "data": None
}
STATUS_CACHE_TTL = 15  # segundos

# =====================
# APLICACIÓN FASTAPI
# =====================

app = FastAPI(
    title="🤖 Agente IA OyP 6.0 - Gateway",
    description="Gateway principal para sistema de IA empresarial",
    version="6.0.1",
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

# Incluir routers
app.include_router(dashboard_routes.router)

# =====================
# WEBSOCKET MANAGER
# =====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {"chat": [], "jobs": []}

    async def connect(self, websocket: WebSocket, group: str):
        await websocket.accept()
        if group not in self.active_connections:
            self.active_connections[group] = []
        self.active_connections[group].append(websocket)
        logger.info(f"Nueva conexión WebSocket al grupo '{group}'. Total en grupo: {len(self.active_connections[group])}")

    def disconnect(self, websocket: WebSocket, group: str):
        if group in self.active_connections and websocket in self.active_connections[group]:
            self.active_connections[group].remove(websocket)
            logger.info(f"Conexión WebSocket cerrada del grupo '{group}'. Total en grupo: {len(self.active_connections[group])}")

    async def broadcast(self, message: dict, group: str):
        if group in self.active_connections:
            for connection in self.active_connections[group][:]:
                try:
                    await connection.send_text(json.dumps(message))
                except WebSocketDisconnect:
                    self.disconnect(connection, group)
                except RuntimeError: # Websocket is closed
                    self.disconnect(connection, group)


manager = ConnectionManager()

# =====================
# RUTAS PRINCIPALES
# =====================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Página principal del dashboard"""
    context = {"request": request, "title": "Dashboard Agente IA"}
    return templates.TemplateResponse("index.html", context)

@app.get("/health")
async def health_check():
    """Health check del gateway"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# =====================
# WEBSOCKET
# =====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, group: str = "chat"):
    """Endpoint WebSocket para 'chat' y 'jobs'."""
    await manager.connect(websocket, group)
    try:
        while True:
            data = await websocket.receive_text()
            # Por ahora, solo hacemos eco de los mensajes al grupo
            # La lógica de proxy se implementará después
            message_to_broadcast = {"group": group, "received_message": data, "timestamp": datetime.now(timezone.utc).isoformat()}
            await manager.broadcast(message_to_broadcast, group)
    except WebSocketDisconnect:
        manager.disconnect(websocket, group)


# =====================
# API DE SERVICIOS (PROXY Y ESTADO)
# =====================

def build_meta(service: str, start_time: float) -> Dict[str, Any]:
    """Construye el diccionario de metadatos para la respuesta."""
    return {
        "service": service,
        "took_ms": int((time.time() - start_time) * 1000),
        "ts": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/services/status")
async def get_services_status():
    """Obtener estado de todos los microservicios con caché de 15s."""
    now = time.time()
    if now - status_cache["timestamp"] < STATUS_CACHE_TTL and status_cache["data"]:
        logger.info("Sirviendo estado de servicios desde caché.")
        return status_cache["data"]

    logger.info("Consultando estado de servicios (sin caché).")
    tasks = []
    for service_name, url in SERVICES_CONFIG.items():
        async def check(name, base_url):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    health_url = f"{base_url}/health"
                    start = time.time()
                    response = await client.get(health_url)
                    took_ms = int((time.time() - start) * 1000)
                    response.raise_for_status()
                    return {"name": name, "status": "online", "took_ms": took_ms}
            except Exception as e:
                return {"name": name, "status": "offline", "error": str(e)}
        tasks.append(check(service_name, url))
    
    results = await asyncio.gather(*tasks)
    
    response_data = {
        "ok": True,
        "data": results,
        "meta": {
            "service": "gateway",
            "took_ms": int((time.time() - now) * 1000),
            "ts": datetime.now(timezone.utc).isoformat()
        }
    }
    
    status_cache["timestamp"] = now
    status_cache["data"] = response_data
    
    return JSONResponse(content=response_data)

@app.api_route("/api/services/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def gateway_proxy_router(request: Request, service_name: str, path: str):
    """Proxy genérico para todos los microservicios con envoltura de respuesta estándar."""
    start_time = time.time()

    if service_name not in SERVICES_CONFIG:
        meta = build_meta(service_name, start_time)
        error_resp = {"ok": False, "code": "BAD_REQUEST", "message": f"Servicio '{service_name}' no encontrado.", "meta": meta}
        return JSONResponse(status_code=404, content=error_resp)

    # --- Construir la URL y obtener timeout ---
    service_url = f"{SERVICES_CONFIG[service_name]}/{path}"
    timeout = SERVICE_TIMEOUTS.get(service_name, SERVICE_TIMEOUTS["default"])

    # --- Preparar la solicitud al microservicio ---
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'connection', 'accept-encoding']}
    params = request.query_params
    content = await request.body()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=request.method,
                url=service_url,
                headers=headers,
                params=params,
                content=content
            )
        
        # --- Envolver la respuesta del microservicio ---
        meta = build_meta(service_name, start_time)
        
        try:
            upstream_data = response.json()
        except json.JSONDecodeError:
            upstream_data = response.text
        
        # Si el microservicio ya responde con el formato estándar, no lo re-envuelvas.
        if isinstance(upstream_data, dict) and "ok" in upstream_data and "data" in upstream_data:
             # Solo añade/actualiza los metadatos del gateway
            upstream_data.setdefault("meta", {})
            upstream_data["meta"]["gateway"] = meta
            return JSONResponse(status_code=response.status_code, content=upstream_data)

        # Si la respuesta es exitosa pero no está envuelta, la envolvemos.
        if 200 <= response.status_code < 300:
            success_resp = {"ok": True, "data": upstream_data, "meta": meta}
            return JSONResponse(status_code=response.status_code, content=success_resp)
        # Si la respuesta es un error y no está envuelta, la envolvemos.
        else:
            error_resp = {
                "ok": False,
                "code": "UPSTREAM_ERROR",
                "message": f"El servicio '{service_name}' devolvió un error.",
                "data": upstream_data,
                "meta": meta
            }
            return JSONResponse(status_code=response.status_code, content=error_resp)

    except httpx.TimeoutException:
        meta = build_meta(service_name, start_time)
        error_resp = {"ok": False, "code": "TIMEOUT", "message": f"Timeout contactando al servicio '{service_name}'.", "meta": meta}
        return JSONResponse(status_code=504, content=error_resp)

    except httpx.RequestError as e:
        meta = build_meta(service_name, start_time)
        error_resp = {"ok": False, "code": "SERVICE_UNAVAILABLE", "message": f"No se pudo contactar al servicio '{service_name}': {e}", "meta": meta}
        return JSONResponse(status_code=503, content=error_resp)

    except Exception as e:
        logger.error(f"Error inesperado en el proxy: {e}")
        meta = build_meta(service_name, start_time)
        error_resp = {"ok": False, "code": "INTERNAL_SERVER_ERROR", "message": f"Error interno en el gateway: {e}", "meta": meta}
        return JSONResponse(status_code=500, content=error_resp)

# =====================
# MAIN
# =====================

if __name__ == "__main__":
    logger.info("🚀 Iniciando servidor en modo desarrollo...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000, # Puerto del Gateway
        reload=True,
        log_level="info"
    )
