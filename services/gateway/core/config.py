""
Configuration settings for the Gateway service.
"""
import os
from typing import List, Optional, Union
from pydantic import AnyHttpUrl, validator, Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application settings
    PROJECT_NAME: str = "Agente IA OYP Gateway"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # Server settings
    HOST: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("GATEWAY_PORT", "8080"))
    
    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # Default frontend dev server
        "http://localhost:8080",  # Default gateway dev server
    ]
    
    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', 'db', 'oyp.sqlite')}"
    )
    
    # Service URLs
    AI_SERVICE_URL: str = os.getenv("AI_SERVICE_URL", "http://ai:8001")
    DOCS_SERVICE_URL: str = os.getenv("DOCS_SERVICE_URL", "http://docproc:8002")
    ANALYTICS_SERVICE_URL: str = os.getenv("ANALYTICS_SERVICE_URL", "http://analytics:8003")
    REPORTS_SERVICE_URL: str = os.getenv("REPORTS_SERVICE_URL", "http://reports:8004")
    CHAT_SERVICE_URL: str = os.getenv("CHAT_SERVICE_URL", "http://chat:8005")
    
    # Job settings
    JOB_TIMEOUT: int = 3600  # 1 hour in seconds
    JOB_CLEANUP_INTERVAL: int = 3600  # 1 hour in seconds
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Initialize settings
settings = Settings()
