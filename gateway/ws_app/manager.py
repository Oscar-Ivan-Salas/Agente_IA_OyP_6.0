"""
Gestor de conexiones WebSocket.

Este módulo maneja las conexiones WebSocket activas y la distribución
de mensajes a los clientes conectados.
"""
import json
import logging
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import WebSocket
from ..pyd_compat import BaseModel

logger = logging.getLogger(__name__)

class WebSocketMessage(BaseModel):
    """Modelo para mensajes WebSocket."""
    type: str
    data: dict

class ConnectionManager:
    """
    Gestiona las conexiones WebSocket activas.
    """
    
    def __init__(self):
        """Inicializa el gestor de conexiones."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_subscriptions: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Acepta una nueva conexión WebSocket.
        
        Args:
            websocket: La conexión WebSocket del cliente.
            client_id: Identificador único del cliente.
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_subscriptions[client_id] = []
        logger.info(f"Cliente conectado: {client_id}")
    
    def disconnect(self, client_id: str):
        """
        Cierra una conexión WebSocket.
        
        Args:
            client_id: Identificador único del cliente.
        """
        self.active_connections.pop(client_id, None)
        self.connection_subscriptions.pop(client_id, None)
        logger.info(f"Cliente desconectado: {client_id}")
    
    async def subscribe(self, client_id: str, channel: str):
        """
        Suscribe un cliente a un canal.
        
        Args:
            client_id: Identificador único del cliente.
            channel: Nombre del canal al que suscribirse.
        """
        if client_id in self.connection_subscriptions:
            if channel not in self.connection_subscriptions[client_id]:
                self.connection_subscriptions[client_id].append(channel)
                logger.info(f"Cliente {client_id} suscrito a {channel}")
    
    async def unsubscribe(self, client_id: str, channel: str):
        """
        Cancela la suscripción de un cliente a un canal.
        
        Args:
            client_id: Identificador único del cliente.
            channel: Nombre del canal del que darse de baja.
        """
        if client_id in self.connection_subscriptions:
            if channel in self.connection_subscriptions[client_id]:
                self.connection_subscriptions[client_id].remove(channel)
                logger.info(f"Cliente {client_id} canceló suscripción a {channel}")
    
    async def send_personal_message(self, message: WebSocketMessage, client_id: str):
        """
        Envía un mensaje a un cliente específico.
        
        Args:
            message: El mensaje a enviar.
            client_id: Identificador único del destinatario.
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_text(message.json())
            except Exception as e:
                logger.error(f"Error enviando mensaje a {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: WebSocketMessage, channel: Optional[str] = None):
        """
        Difunde un mensaje a todos los clientes suscritos a un canal.
        
        Args:
            message: El mensaje a difundir.
            channel: Canal al que difundir el mensaje. Si es None, se envía a todos los canales.
        """
        disconnected = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                # Verificar si el cliente está suscrito al canal
                if (channel is None or 
                    client_id not in self.connection_subscriptions or 
                    channel in self.connection_subscriptions[client_id]):
                    
                    await websocket.send_text(message.json())
            except Exception as e:
                logger.error(f"Error en difusión a {client_id}: {e}")
                disconnected.append(client_id)
        
        # Eliminar conexiones fallidas
        for client_id in disconnected:
            self.disconnect(client_id)
    
    async def notify_job_update(self, job_id: UUID, job_data: dict):
        """
        Notifica una actualización de trabajo a los clientes suscritos.
        
        Args:
            job_id: ID del trabajo actualizado.
            job_data: Datos actualizados del trabajo.
        """
        message = WebSocketMessage(
            type="job_update",
            data={"job_id": str(job_id), **job_data}
        )
        await self.broadcast(message, f"jobs:{job_id}")
    
    async def notify_system_message(self, title: str, message: str, level: str = "info"):
        """
        Envía un mensaje del sistema a los clientes.
        
        Args:
            title: Título del mensaje.
            message: Contenido del mensaje.
            level: Nivel del mensaje (info, warning, error, success).
        """
        message = WebSocketMessage(
            type="system_message",
            data={
                "title": title,
                "message": message,
                "level": level,
                "timestamp": str(datetime.utcnow())
            }
        )
        await self.broadcast(message, "system:notifications")
