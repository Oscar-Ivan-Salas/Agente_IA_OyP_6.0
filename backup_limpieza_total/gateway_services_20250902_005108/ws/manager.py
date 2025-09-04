"""
WebSocket connection manager for real-time communication.
"""
import json
import asyncio
from typing import Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
from ..core.config import settings

class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}
        self.lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Register a new WebSocket connection."""
        await websocket.accept()
        async with self.lock:
            self.active_connections[client_id] = websocket
            self.subscriptions[client_id] = []
            
    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        async with self.lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]
                
    async def subscribe(self, client_id: str, channel: str):
        """Subscribe a client to a channel."""
        async with self.lock:
            if client_id in self.active_connections:
                if channel not in self.subscriptions[client_id]:
                    self.subscriptions[client_id].append(channel)
                    
    async def unsubscribe(self, client_id: str, channel: str):
        """Unsubscribe a client from a channel."""
        async with self.lock:
            if client_id in self.subscriptions and channel in self.subscriptions[client_id]:
                self.subscriptions[client_id].remove(channel)
                
    async def send_personal_message(self, message: dict, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json({"ok": True, "data": message})
            except WebSocketDisconnect:
                await self.disconnect(client_id)
                
    async def broadcast(self, message: dict, channel: str):
        """Broadcast a message to all clients subscribed to a channel."""
        disconnected_clients = []
        
        async with self.lock:
            for client_id, websocket in list(self.active_connections.items()):
                if channel in self.subscriptions.get(client_id, []):
                    try:
                        await websocket.send_json({"ok": True, "data": message})
                    except WebSocketDisconnect:
                        disconnected_clients.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected_clients:
                await self.disconnect(client_id)
                
    async def send_job_update(self, job_id: str, status: str, progress: int = 0, message: str = ""):
        """Send a job update to all subscribers of this job."""
        update = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.broadcast(update, f"job_updates:{job_id}")

# Global WebSocket manager instance
websocket_manager = ConnectionManager()
