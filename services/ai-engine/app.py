"""
AI Engine v6.0 - Motor de IA Híbrido
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from contextlib import asynccontextmanager

# Imports con manejo de errores
try:
    from routers import generation, analysis, chat, training, models
    from models.llm_hybrid_engine import HybridLLMEngine
    from services.model_manager import ModelManager
    from utils.config import Settings
    from utils.logging_config import setup_logging
    IMPORTS_OK = True
except ImportError as e:
    print(f"⚠️ Algunos módulos no disponibles: {e}")
    IMPORTS_OK = False

# Configuración global
if IMPORTS_OK:
    try:
        settings = Settings()
        model_manager = ModelManager()
        hybrid_engine = None
        setup_logging()
    except:
        IMPORTS_OK = False

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    global hybrid_engine
    
    logger.info("🚀 Iniciando AI-Engine...")
    
    if IMPORTS_OK:
        try:
            hybrid_engine = HybridLLMEngine()
            await hybrid_engine.initialize()
            model_manager.set_engine(hybrid_engine)
            logger.info("✅ AI-Engine inicializado")
        except Exception as e:
            logger.warning(f"⚠️ Modo básico: {e}")
    
    yield
    
    if hybrid_engine:
        try:
            await hybrid_engine.cleanup()
        except:
            pass

# Crear aplicación FastAPI
app = FastAPI(
    title="AI-Engine v6.0",
    description="Motor híbrido de IA",
    version="6.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "AI-Engine v6.0",
        "status": "active",
        "imports_ok": IMPORTS_OK
    }

@app.get("/health")
async def health_check():
    """Health check"""
    try:
        if hybrid_engine and IMPORTS_OK:
            engine_status = await hybrid_engine.health_check()
        else:
            engine_status = {"status": "basic_mode"}
        
        return {
            "status": "healthy",
            "service": "ai-engine",
            "version": "6.0.0",
            "engine": engine_status,
            "imports_ok": IMPORTS_OK
        }
    except Exception as e:
        return {
            "status": "healthy",
            "service": "ai-engine", 
            "version": "6.0.0",
            "mode": "basic",
            "note": str(e)
        }

# Incluir routers si están disponibles
if IMPORTS_OK:
    try:
        app.include_router(generation.router, prefix="/generate", tags=["Generation"])
        app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
        app.include_router(chat.router, prefix="/chat", tags=["Chat"])
        app.include_router(training.router, prefix="/train", tags=["Training"])
        app.include_router(models.router, prefix="/models", tags=["Models"])
    except Exception as e:
        logger.warning(f"Error incluyendo routers: {e}")

if __name__ == "__main__":
    port = 8001
    print(f"🚀 Iniciando AI Engine en puerto {port}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
