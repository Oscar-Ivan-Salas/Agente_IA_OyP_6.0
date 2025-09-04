"""
Configuración del Gateway.

Este módulo maneja la configuración de la aplicación,
cargando valores desde variables de entorno con valores por defecto.
"""
import os
from typing import List, Optional
from gateway.pyd_compat import BaseSettings, Field, AnyHttpUrl

class Settings(BaseSettings):
    # Configuración básica
    DEBUG: bool = Field(
        default=os.getenv("DEBUG", "False").lower() in ("true", "1", "t"),
        description="Modo de depuración"
    )
    
    ENVIRONMENT: str = Field(
        default=os.getenv("ENVIRONMENT", "development"),
        description="Entorno de ejecución (development, staging, production)"
    )
    
    # Configuración del servidor
    HOST: str = Field(
        default=os.getenv("HOST", "0.0.0.0"),
        description="Dirección IP en la que escucha el servidor"
    )
    
    PORT: int = Field(
        default=int(os.getenv("PORT", "8000")),
        description="Puerto en el que escucha el servidor"
    )
    
    WORKERS: int = Field(
        default=int(os.getenv("WORKERS", "1")),
        description="Número de workers para el servidor ASGI"
    )
    
    # Configuración de CORS
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",  # React frontend
            "http://localhost:8080",  # Vue frontend
        ]
    )
    
    # Configuración de base de datos
    DATABASE_URL: str = Field(
        default=os.getenv("DATABASE_URL", "sqlite:///./agente_ia.db"),
        description="URL de conexión a la base de datos"
    )
    
    # Configuración de autenticación
    SECRET_KEY: str = Field(
        default=os.getenv("SECRET_KEY", "your-secret-key-change-in-production"),
        description="Clave secreta para firmar tokens JWT"
    )
    
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")),  # 24 horas
        description="Tiempo de expiración del token de acceso en minutos"
    )
    
    # Configuración de servicios
    AI_ENGINE_URL: str = Field(
        default=os.getenv("AI_ENGINE_URL", "http://ai-engine:8001"),
        description="URL del servicio de motor de IA"
    )
    
    DOC_PROCESSOR_URL: str = Field(
        default=os.getenv("DOC_PROCESSOR_URL", "http://document-processor:8002"),
        description="URL del servicio de procesamiento de documentos"
    )
    
    ANALYTICS_URL: str = Field(
        default=os.getenv("ANALYTICS_URL", "http://analytics:8003"),
        description="URL del servicio de análisis"
    )
    
    REPORTS_URL: str = Field(
        default=os.getenv("REPORTS_URL", "http://reports:8004"),
        description="URL del servicio de reportes"
    )
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Instancia de configuración
settings = Settings()
