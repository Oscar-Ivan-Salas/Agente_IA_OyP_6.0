"""
Router training
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def training_root():
    return {"service": "training", "status": "ready"}
