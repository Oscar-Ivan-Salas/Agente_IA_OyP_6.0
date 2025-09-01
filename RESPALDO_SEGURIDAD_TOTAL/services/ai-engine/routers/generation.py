"""
Router de generación
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class GenerationRequest(BaseModel):
    prompt: str
    model_preference: str = "auto"
    max_length: int = 512
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    response: str
    model_used: str
    model_type: str
    tokens_used: int
    inference_time: float
    timestamp: str
    engine: str

async def get_engine():
    """Obtener motor"""
    from app import hybrid_engine
    if not hybrid_engine or not hybrid_engine.is_ready():
        raise HTTPException(status_code=503, detail="AI Engine no disponible")
    return hybrid_engine

@router.post("/text", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, engine = Depends(get_engine)):
    """Generar texto"""
    try:
        result = await engine.generate_text(
            prompt=request.prompt,
            model_preference=request.model_preference,
            max_length=request.max_length,
            temperature=request.temperature
        )
        return GenerationResponse(**result)
    except Exception as e:
        logger.error(f"Error generación: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_models(engine = Depends(get_engine)):
    """Modelos disponibles"""
    try:
        models = await engine.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
