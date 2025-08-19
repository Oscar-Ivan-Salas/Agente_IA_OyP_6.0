"""
Procesador de Documentos - Servicio Principal
Puerto: 8002
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path

app = FastAPI(
    title="Procesador de Documentos",
    description="Microservicio del Agente IA OyP 6.0",
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

@app.get("/")
async def root():
    return {
        "message": "Bienvenido a Procesador de Documentos",
        "service": "document-processor",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "document-processor",
        "port": 8002
    }

@app.get("/info")
async def service_info():
    return {
        "name": "document-processor",
        "description": "Procesador de Documentos",
        "port": 8002,
        "endpoints": ["/", "/health", "/info"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True
    )
