# services/gateway/core/config.py
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


# Carga .env desde la raíz del repo (sube hasta encontrarlo)
def _load_env():
    here = Path(__file__).resolve()
    for _ in range(6):
        env_path = here.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return
        here = here.parent


_load_env()


class Settings(BaseModel):
    # Application settings
    PROJECT_NAME: str = "Agente IA OYP Gateway"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS settings
    CORS_ORIGINS: list[str] = ["*"]
    
    # Database settings
    DB_URL: str = "sqlite:///./db.sqlite3"
    
    # Service URLs
    AI_SERVICE_URL: str = "http://ai:8001"
    DOCS_SERVICE_URL: str = "http://docproc:8002"
    ANALYTICS_SERVICE_URL: str = "http://analytics:8003"
    REPORTS_SERVICE_URL: str = "http://reports:8004"
    CHAT_SERVICE_URL: str = "http://chat:8005"
    
    # Job settings
    JOB_TIMEOUT: int = 3600  # 1 hour in seconds
    JOB_CLEANUP_INTERVAL: int = 3600  # 1 hour in seconds

    @classmethod
    def from_env(cls) -> "Settings":
        cors_raw = os.getenv("CORS_ORIGINS", "*").strip()
        if cors_raw == "*":
            cors = ["*"]
        else:
            try:
                cors = json.loads(cors_raw)  # e.g.: '["http://localhost:3000"]'
                if not isinstance(cors, list):
                    raise ValueError
            except Exception:
                cors = [x.strip() for x in cors_raw.split(",") if x.strip()]
        
        return cls(
            PROJECT_NAME=os.getenv("PROJECT_NAME", cls().PROJECT_NAME),
            VERSION=os.getenv("VERSION", cls().VERSION),
            ENVIRONMENT=os.getenv("ENVIRONMENT", cls().ENVIRONMENT),
            DEBUG=os.getenv("DEBUG", str(cls().DEBUG)).lower() in ("1", "true", "yes", "on"),
            HOST=os.getenv("HOST", cls().HOST),
            PORT=int(os.getenv("PORT", str(cls().PORT))),
            SECRET_KEY=os.getenv("SECRET_KEY", cls().SECRET_KEY),
            CORS_ORIGINS=cors,
            DB_URL=os.getenv("DB_URL", cls().DB_URL),
            AI_SERVICE_URL=os.getenv("AI_SERVICE_URL", cls().AI_SERVICE_URL),
            DOCS_SERVICE_URL=os.getenv("DOCS_SERVICE_URL", cls().DOCS_SERVICE_URL),
            ANALYTICS_SERVICE_URL=os.getenv("ANALYTICS_SERVICE_URL", cls().ANALYTICS_SERVICE_URL),
            REPORTS_SERVICE_URL=os.getenv("REPORTS_SERVICE_URL", cls().REPORTS_SERVICE_URL),
            CHAT_SERVICE_URL=os.getenv("CHAT_SERVICE_URL", cls().CHAT_SERVICE_URL),
            JOB_TIMEOUT=int(os.getenv("JOB_TIMEOUT", str(cls().JOB_TIMEOUT))),
            JOB_CLEANUP_INTERVAL=int(os.getenv("JOB_CLEANUP_INTERVAL", str(cls().JOB_CLEANUP_INTERVAL))),
        )


settings = Settings.from_env()
