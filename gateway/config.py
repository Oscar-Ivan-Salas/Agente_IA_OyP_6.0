""
Configuración del Gateway.

Este módulo maneja la configuración de la aplicación,
cargando valores desde variables de entorno con valores por defecto.
"""
import os
from typing import List
from pydantic import BaseSettings, AnyHttpUrl

class Settings(BaseSettings):
    # Configuración básica
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Configuración del servidor
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    # Configuración de CORS
    CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # React frontend
        "http://localhost:8080",  # Vue frontend
    ]
    
    # Configuración de base de datos
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "sqlite:///./agente_ia.db"
    )
    
    # Configuración de autenticación
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 horas
    
    # Configuración de servicios
    AI_ENGINE_URL: str = os.getenv("AI_ENGINE_URL", "http://ai-engine:8001")
    DOC_PROCESSOR_URL: str = os.getenv("DOC_PROCESSOR_URL", "http://document-processor:8002")
    ANALYTICS_URL: str = os.getenv("ANALYTICS_URL", "http://analytics:8003")
    REPORTS_URL: str = os.getenv("REPORTS_URL", "http://reports:8004")
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Instancia de configuración
settings = Settings()
