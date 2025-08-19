"""
Router chat
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def chat_root():
    return {"service": "chat", "status": "ready"}
