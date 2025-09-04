#!/usr/bin/env python3
"""
🚀 AGENTE IA OYP 6.0 - AI ENGINE
=================================
Microservicio para Modelos de Lenguaje y Análisis de Texto.
Archivo: services/ai-engine/app.py
"""

import httpx
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# =====================
# CONFIGURACIÓN GLOBAL
# =====================

# Cargar variables de entorno desde .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_ENABLED = False

# Modelos locales preconfigurados
LOCAL_MODELS = {
    "llama": {"model_name": "llama3.1", "status": "unavailable"},
    "r1": {"model_name": "deepseek-coder-v2", "status": "unavailable"},
    "bert": {"model_name": "nomic-embed-text", "status": "unavailable"} # Modelo de embedding
}

# =====================
# APLICACIÓN FASTAPI
# =====================

app = FastAPI(
    title="🤖 AI Engine Service",
    description="Microservicio para interactuar con modelos de IA locales y en la nube.",
    version="1.0.0"
)

# =====================
# MODELOS PYDANTIC (CONTRATOS DE API)
# =====================

class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Texto a analizar.")

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Texto a resumir.")
    max_tokens: Optional[int] = Field(150, description="Máximo de tokens para el resumen.")

class ChatRequest(BaseModel):
    message: str = Field(..., description="Mensaje del usuario.")
    model: str = Field("llama", description="Modelo a utilizar ('llama' o 'r1').")
    history: Optional[List[Dict[str, str]]] = Field([], description="Historial de la conversación.")

# =====================
# UTILIDADES Y HELPERS
# =====================

def build_meta(start_time: float) -> Dict[str, Any]:
    """Construye el diccionario de metadatos para la respuesta."""
    return {
        "took_ms": int((time.time() - start_time) * 1000),
        "ts": datetime.now(timezone.utc).isoformat()
    }

def create_response(data: Any, ok: bool = True, start_time: float = time.time()) -> JSONResponse:
    """Crea una respuesta JSON estandarizada."""
    content = {
        "ok": ok,
        "data": data,
        "meta": build_meta(start_time)
    }
    return JSONResponse(content=content)

def create_error_response(code: str, message: str, status_code: int = 500, start_time: float = time.time()) -> JSONResponse:
    """Crea una respuesta de error JSON estandarizada."""
    content = {
        "ok": False,
        "error": {"code": code, "message": message},
        "meta": build_meta(start_time)
    }
    return JSONResponse(status_code=status_code, content=content)

async def check_ollama_models():
    """Verifica la disponibilidad de los modelos en Ollama."""
    global OLLAMA_ENABLED
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=5.0) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            OLLAMA_ENABLED = True
            available_models = [m['name'].split(':')[0] for m in response.json().get('models', [])]
            for key, model_info in LOCAL_MODELS.items():
                if model_info["model_name"] in available_models:
                    LOCAL_MODELS[key]["status"] = "available"
            logger.info(f"Ollama conectado. Modelos disponibles: {[k for k, v in LOCAL_MODELS.items() if v['status'] == 'available']}")
    except Exception as e:
        OLLAMA_ENABLED = False
        logger.warning(f"No se pudo conectar a Ollama en {OLLAMA_URL}. El servicio funcionará en modo stub. Error: {e}")

async def ollama_chat_completion(model: str, prompt: str, history: List[Dict] = []) -> Dict:
    """Llama al endpoint de chat de Ollama."""
    messages = history + [{"role": "user", "content": prompt}]
    model_name = LOCAL_MODELS.get(model, {}).get("model_name", "llama3.1")
    
    async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=25.0) as client:
        response = await client.post("/api/chat", json={
            "model": model_name,
            "messages": messages,
            "stream": False
        })
        response.raise_for_status()
        return response.json()

# =====================
# ENDPOINTS DE LA API
# =====================

@app.on_event("startup")
async def startup_event():
    await check_ollama_models()

@app.get("/health")
async def health_check():
    """Endpoint de salud para el Gateway."""
    return create_response({"status": "healthy"})

@app.get("/models")
async def get_models():
    """Lista los modelos de IA disponibles y su estado."""
    start_time = time.time()
    # Simplificamos la respuesta para que sea más clara
    models_status = {k: v["status"] for k, v in LOCAL_MODELS.items()}
    return create_response(models_status, start_time=start_time)

@app.post("/analyze_text")
async def analyze_text(req: AnalyzeTextRequest):
    """Analiza texto para extraer sentimiento, idioma y frases clave."""
    start_time = time.time()
    if not OLLAMA_ENABLED:
        # Respuesta Stub
        stub_data = {
            "sentiment": "neutral",
            "lang": "es",
            "keyphrases": ["contrato", "servicio", "lima"],
            "summary": "Este es un resumen simulado del texto proporcionado.",
            "simulated": True
        }
        return create_response(stub_data, start_time=start_time)

    try:
        prompt = f"Analiza el siguiente texto. Extrae el sentimiento (positivo, negativo, neutral), el idioma (código ISO 639-1), 5 frases clave (keyphrases) y un resumen corto (summary) de 1 frase. Formatea la salida como un JSON con las claves 'sentiment', 'lang', 'keyphrases', y 'summary'. Texto: \n\n{req.text}"
        response = await ollama_chat_completion("llama", prompt)
        analysis = json.loads(response['message']['content'])
        return create_response(analysis, start_time=start_time)
    except Exception as e:
        return create_error_response("ANALYSIS_FAILED", f"Error al analizar el texto: {e}", start_time=start_time)

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Genera un resumen de un texto largo."""
    start_time = time.time()
    if not OLLAMA_ENABLED:
        return create_response({"summary": "Este es un resumen simulado.", "simulated": True}, start_time=start_time)

    try:
        prompt = f"Resume el siguiente texto en un máximo de {req.max_tokens} tokens: \n\n{req.text}"
        response = await ollama_chat_completion("llama", prompt)
        summary = response['message']['content']
        return create_response({"summary": summary}, start_time=start_time)
    except Exception as e:
        return create_error_response("SUMMARIZE_FAILED", f"Error al generar el resumen: {e}", start_time=start_time)

@app.post("/chat")
async def chat(req: ChatRequest):
    """Maneja una solicitud de chat, manteniendo el historial."""
    start_time = time.time()
    if not OLLAMA_ENABLED:
        latency = int((time.time() - start_time) * 1000)
        return create_response({"reply": "Respuesta simulada del modelo de chat.", "latency_ms": latency, "simulated": True}, start_time=start_time)

    try:
        response = await ollama_chat_completion(req.model, req.message, req.history)
        reply = response['message']['content']
        latency = int((time.time() - start_time) * 1000)
        return create_response({"reply": reply, "latency_ms": latency}, start_time=start_time)
    except Exception as e:
        return create_error_response("CHAT_FAILED", f"Error en la comunicación con el modelo: {e}", start_time=start_time)

@app.post("/configure")
async def configure_service():
    """Endpoint stub para configuración futura."""
    return create_error_response("NOT_IMPLEMENTED", "La configuración dinámica aún no está implementada.", status_code=501)

# =====================
# MAIN
# =====================

if __name__ == "__main__":
    logger.info("🚀 Iniciando AI Engine Service...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )