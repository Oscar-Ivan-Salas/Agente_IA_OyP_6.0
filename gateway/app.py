"""
API Gateway - Agente IA OyP 6.0
Puerto: 8080
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import os
from pathlib import Path

app = FastAPI(
    title="Agente IA OyP 6.0 - Gateway",
    description="API Gateway principal del sistema",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar archivos estáticos y templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# URLs de microservicios
SERVICES = {
    "ai-engine": "http://localhost:8001",
    "document-processor": "http://localhost:8002", 
    "analytics-engine": "http://localhost:8003",
    "report-generator": "http://localhost:8004"
}

@app.get("/")
async def dashboard(request: Request):
    """Dashboard principal"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Agente IA OyP 6.0",
        "services": SERVICES
    })

@app.get("/health")
async def health_check():
    """Health check del gateway"""
    return {
        "status": "healthy",
        "service": "gateway",
        "port": 8080
    }

@app.get("/services/status")
async def services_status():
    """Estado de todos los microservicios"""
    status = {}
    
    async with httpx.AsyncClient() as client:
        for service_name, service_url in SERVICES.items():
            try:
                response = await client.get(f"{service_url}/health", timeout=5.0)
                status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "url": service_url
                }
            except Exception as e:
                status[service_name] = {
                    "status": "offline",
                    "error": str(e),
                    "url": service_url
                }
    
    return status

# Proxy endpoints para microservicios
@app.api_route("/api/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_service(service_name: str, path: str, request: Request):
    """Proxy a microservicios"""
    
    if service_name not in SERVICES:
        return {"error": f"Servicio {service_name} no encontrado"}
    
    service_url = SERVICES[service_name]
    url = f"{service_url}/{path}"
    
    async with httpx.AsyncClient() as client:
        try:
            # Reenviar request al microservicio
            response = await client.request(
                method=request.method,
                url=url,
                params=request.query_params,
                content=await request.body(),
                headers=dict(request.headers)
            )
            return response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        except Exception as e:
            return {"error": f"Error conectando con {service_name}: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
