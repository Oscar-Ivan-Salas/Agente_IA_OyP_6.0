"""
Chat API endpoints.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class ChatMessage(BaseModel):
    session_id: str
    message: str
    client: str
    project: str
    mode: str = "retrieval_only"
    k: int = 4

@router.post("/session")
async def create_session():
    """Create a new chat session."""
    return {"ok": True, "data": {"session_id": "session_123"}}

@router.post("/message")
async def send_message(message: ChatMessage):
    """Send a chat message."""
    return {
        "ok": True,
        "data": {
            "reply": f"Echo: {message.message}",
            "citations": [],
            "mode": message.mode
        }
    }

@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "ok": True,
                "data": {
                    "reply": f"Echo: {data}",
                    "citations": [],
                    "mode": "websocket"
                }
            })
    except WebSocketDisconnect:
        pass
