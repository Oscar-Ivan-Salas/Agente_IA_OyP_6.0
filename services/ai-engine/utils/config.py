"""
Configuración del sistema
"""

import os

class Settings:
    def __init__(self):
        # General
        self.app_name = os.getenv("APP_NAME", "AI-Engine v6.0")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # APIs
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Configuración
        self.enable_local_models = os.getenv("ENABLE_LOCAL_MODELS", "true").lower() == "true"
        self.enable_cloud_models = os.getenv("ENABLE_CLOUD_MODELS", "true").lower() == "true"
        
        # BD y Cache
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./ai_engine.db")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
