"""
Router models
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def models_root():
    return {"service": "models", "status": "ready"}
