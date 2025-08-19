"""
Motor de Analytics - Servicio Principal
Puerto: 8003
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path

app = FastAPI(
    title="Motor de Analytics",
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
        "message": "Bienvenido a Motor de Analytics",
        "service": "analytics-engine",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "analytics-engine",
        "port": 8003
    }

@app.get("/info")
async def service_info():
    return {
        "name": "analytics-engine",
        "description": "Motor de Analytics",
        "port": 8003,
        "endpoints": ["/", "/health", "/info"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True
    )
