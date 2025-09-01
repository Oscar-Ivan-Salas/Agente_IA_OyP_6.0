"""
Router de análisis
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class AnalysisRequest(BaseModel):
    text: str
    analysis_types: List[str] = ["sentiment"]

class AnalysisResponse(BaseModel):
    text_length: int
    word_count: int
    analysis_results: Dict[str, Any]
    timestamp: str

async def get_engine():
    """Obtener motor"""
    from app import hybrid_engine
    if not hybrid_engine or not hybrid_engine.is_ready():
        raise HTTPException(status_code=503, detail="AI Engine no disponible")
    return hybrid_engine

@router.post("/text", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest, engine = Depends(get_engine)):
    """Analizar texto"""
    try:
        results = {}
        
        if "sentiment" in request.analysis_types:
            sentiment = await engine.analyze_sentiment(request.text)
            results["sentiment"] = sentiment
        
        if "summary" in request.analysis_types:
            summary = await engine.summarize_text(request.text)
            results["summary"] = summary
        
        return AnalysisResponse(
            text_length=len(request.text),
            word_count=len(request.text.split()),
            analysis_results=results,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error análisis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
