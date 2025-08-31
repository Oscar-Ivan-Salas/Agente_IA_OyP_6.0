#!/usr/bin/env python3
"""
⚙️ CONFIGURACIÓN DEL GATEWAY - Agente IA OyP 6.0
================================================

Configuración centralizada del API Gateway.
Gestiona todas las variables de entorno, URLs de servicios y parámetros del sistema.

Líneas: ~85
"""

import os
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import logging

class GatewaySettings(BaseSettings):
    """Configuración principal del Gateway"""
    
    # ===============================================
    # CONFIGURACIÓN BÁSICA
    # ===============================================
    
    app_name: str = Field(default="Agente IA OyP 6.0", env="APP_NAME")
    version: str = Field(default="1.0.0", env="VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # ===============================================
    # SERVIDOR Y PUERTOS
    # ===============================================
    
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="GATEWAY_PORT")
    reload: bool = Field(default=True, env="RELOAD")
    workers: int = Field(default=1, env="WORKERS")
    
    # ===============================================
    # MICROSERVICIOS - URLs COMPLETAS
    # ===============================================
    
    ai_engine_url: str = Field(default="http://localhost:8001", env="AI_ENGINE_URL")
    document_processor_url: str = Field(default="http://localhost:8002", env="DOCUMENT_PROCESSOR_URL")
    analytics_engine_url: str = Field(default="http://localhost:8003", env="ANALYTICS_ENGINE_URL")
    report_generator_url: str = Field(default="http://localhost:8004", env="REPORT_GENERATOR_URL")
    chat_ai_service_url: str = Field(default="http://localhost:8005", env="CHAT_AI_SERVICE_URL")
    
    # ===============================================
    # BASE DE DATOS
    # ===============================================
    
    database_url: str = Field(default="sqlite:///./gateway.db", env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # ===============================================
    # APIS EXTERNAS (OPCIONAL)
    # ===============================================
    
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    # ===============================================
    # SEGURIDAD
    # ===============================================
    
    secret_key: str = Field(default="gateway-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # ===============================================
    # CORS
    # ===============================================
    
    cors_origins: List[str] = Field(
        default=["http://localhost:8000", "http://127.0.0.1:8000"],
        env="CORS_ORIGINS"
    )
    cors_methods: List[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    # ===============================================
    # TIMEOUTS Y LÍMITES
    # ===============================================
    
    http_timeout: int = Field(default=30, env="HTTP_TIMEOUT")
    websocket_timeout: int = Field(default=60, env="WEBSOCKET_TIMEOUT")
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    max_connections: int = Field(default=100, env="MAX_CONNECTIONS")
    
    # ===============================================
    # LOGGING
    # ===============================================
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="gateway.log", env="LOG_FILE")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def services_config(self) -> Dict[str, Dict[str, str]]:
        """Configuración de todos los microservicios"""
        return {
            "ai_engine": {
                "name": "AI Engine",
                "url": self.ai_engine_url,
                "health_endpoint": "/health",
                "description": "Motor de Inteligencia Artificial"
            },
            "document_processor": {
                "name": "Document Processor", 
                "url": self.document_processor_url,
                "health_endpoint": "/health",
                "description": "Procesador de Documentos"
            },
            "analytics_engine": {
                "name": "Analytics Engine",
                "url": self.analytics_engine_url,
                "health_endpoint": "/health", 
                "description": "Motor de Análisis Estadístico"
            },
            "report_generator": {
                "name": "Report Generator",
                "url": self.report_generator_url,
                "health_endpoint": "/health",
                "description": "Generador de Reportes"
            },
            "chat_ai_service": {
                "name": "Chat AI Service",
                "url": self.chat_ai_service_url,
                "health_endpoint": "/health",
                "description": "Servicio de Chat con IA"
            }
        }

    def setup_logging(self) -> None:
        """Configurar logging del sistema"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format=self.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_file)
            ]
        )
        
        # Configurar loggers específicos
        loggers = [
            "uvicorn",
            "fastapi", 
            "httpx",
            "gateway"
        ]
        
        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, self.log_level.upper()))

# ===============================================
# INSTANCIA GLOBAL DE CONFIGURACIÓN
# ===============================================

settings = GatewaySettings()

# Configurar logging al importar
settings.setup_logging()

# Logger para este módulo
logger = logging.getLogger("gateway.settings")
logger.info(f"🚀 Configuración cargada: {settings.app_name} v{settings.version}")
logger.info(f"🌐 Entorno: {settings.environment}")
logger.info(f"🔧 Debug: {settings.debug}")
logger.info(f"📡 Servicios configurados: {len(settings.services_config)}")