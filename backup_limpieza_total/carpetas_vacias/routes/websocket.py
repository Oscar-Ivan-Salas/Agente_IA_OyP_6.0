#!/usr/bin/env python3
"""
🚀 AGENTE IA OYP 6.0 - WEBSOCKET HANDLER
======================================
Manejo avanzado de WebSocket para tiempo real
Archivo: gateway/routes/websocket.py (400 líneas completas)
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect, APIRouter, HTTPException
from fastapi.websockets import WebSocketState
import httpx

# =====================
# CONFIGURACIÓN
# =====================

logger = logging.getLogger(__name__)

# Router para WebSocket
router = APIRouter()

# =====================
# ENUMS Y MODELOS
# =====================

class MessageType(str, Enum):
    CHAT = "chat"
    SYSTEM = "system"  
    NOTIFICATION = "notification"
    STATUS_UPDATE = "status_update"
    ANALYTICS_UPDATE = "analytics_update"
    TRAINING_UPDATE = "training_update"
    AGENT_ACTION = "agent_action"
    ERROR = "error"

class ConnectionStatus(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class WebSocketMessage:
    id: str
    type: MessageType
    content: Any
    timestamp: str
    user_id: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class ConnectionInfo:
    websocket: WebSocket
    user_id: str
    connection_time: datetime
    last_ping: datetime
    subscriptions: Set[str]
    status: ConnectionStatus

# =====================
# WEBSOCKET MANAGER
# =====================

class AdvancedWebSocketManager:
    def __init__(self):
        # Conexiones activas: {connection_id: ConnectionInfo}
        self.active_connections: Dict[str, ConnectionInfo] = {}
        
        # Historial de mensajes por tipo
        self.message_history: Dict[MessageType, List[WebSocketMessage]] = {
            message_type: [] for message_type in MessageType
        }
        
        # Suscripciones: {topic: {connection_ids}}
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # Métricas
        self.metrics = {
            "total_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0
        }
        
        # Configuración
        self.max_history_per_type = 100
        self.ping_interval = 30  # segundos
        self.connection_timeout = 300  # segundos
        
        # Iniciar tarea de limpieza
        asyncio.create_task(self._cleanup_task())
        asyncio.create_task(self._ping_task())

    async def connect(self, websocket: WebSocket, user_id: str = None) -> str:
        """Conectar nuevo WebSocket"""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        user_id = user_id or f"user_{connection_id[:8]}"
        
        connection_info = ConnectionInfo(
            websocket=websocket,
            user_id=user_id,
            connection_time=datetime.now(),
            last_ping=datetime.now(),
            subscriptions=set(),
            status=ConnectionStatus.CONNECTED
        )
        
        self.active_connections[connection_id] = connection_info
        self.metrics["total_connections"] += 1
        
        logger.info(f"Nueva conexión WebSocket: {connection_id} (user: {user_id})")
        
        # Enviar mensaje de bienvenida
        welcome_message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM,
            content={
                "message": "Conectado al Agente IA OyP 6.0",
                "connection_id": connection_id,
                "user_id": user_id,
                "features": ["chat", "notifications", "real_time_updates"]
            },
            timestamp=datetime.now().isoformat()
        )
        
        await self._send_message_to_connection(connection_id, welcome_message)
        
        # Enviar historial reciente
        await self._send_recent_history(connection_id)
        
        return connection_id

    async def disconnect(self, connection_id: str):
        """Desconectar WebSocket"""
        if connection_id in self.active_connections:
            connection_info = self.active_connections[connection_id]
            
            # Remover de suscripciones
            for topic in connection_info.subscriptions:
                if topic in self.subscriptions:
                    self.subscriptions[topic].discard(connection_id)
                    if not self.subscriptions[topic]:
                        del self.subscriptions[topic]
            
            # Remover conexión
            del self.active_connections[connection_id]
            
            logger.info(f"Conexión WebSocket cerrada: {connection_id}")

    async def send_message(self, message: WebSocketMessage, target_connections: List[str] = None):
        """Enviar mensaje a conexiones específicas o broadcast"""
        # Agregar a historial
        self._add_to_history(message)
        
        if target_connections:
            # Enviar a conexiones específicas
            for connection_id in target_connections:
                await self._send_message_to_connection(connection_id, message)
        else:
            # Broadcast a todas las conexiones
            await self.broadcast(message)

    async def broadcast(self, message: WebSocketMessage):
        """Broadcast mensaje a todas las conexiones activas"""
        self._add_to_history(message)
        
        disconnected = []
        for connection_id, connection_info in self.active_connections.items():
            try:
                await self._send_message_to_connection(connection_id, message)
            except Exception as e:
                logger.error(f"Error enviando a {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Limpiar conexiones desconectadas
        for connection_id in disconnected:
            await self.disconnect(connection_id)

    async def subscribe(self, connection_id: str, topic: str):
        """Suscribir conexión a un tópico"""
        if connection_id in self.active_connections:
            self.active_connections[connection_id].subscriptions.add(topic)
            
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            self.subscriptions[topic].add(connection_id)
            
            logger.info(f"Conexión {connection_id} suscrita a {topic}")

    async def unsubscribe(self, connection_id: str, topic: str):
        """Desuscribir conexión de un tópico"""
        if connection_id in self.active_connections:
            self.active_connections[connection_id].subscriptions.discard(topic)
            
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(connection_id)
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]

    async def publish_to_topic(self, topic: str, message: WebSocketMessage):
        """Publicar mensaje a todas las conexiones suscritas a un tópico"""
        if topic in self.subscriptions:
            subscribers = list(self.subscriptions[topic])
            await self.send_message(message, subscribers)

    async def _send_message_to_connection(self, connection_id: str, message: WebSocketMessage):
        """Enviar mensaje a una conexión específica"""
        if connection_id not in self.active_connections:
            return
        
        connection_info = self.active_connections[connection_id]
        
        try:
            if connection_info.websocket.client_state == WebSocketState.CONNECTED:
                await connection_info.websocket.send_text(json.dumps(asdict(message)))
                self.metrics["messages_sent"] += 1
            else:
                await self.disconnect(connection_id)
        except Exception as e:
            logger.error(f"Error enviando mensaje a {connection_id}: {e}")
            await self.disconnect(connection_id)

    async def _send_recent_history(self, connection_id: str):
        """Enviar historial reciente a nueva conexión"""
        # Enviar últimos mensajes de chat
        recent_chat = self.message_history[MessageType.CHAT][-10:]
        for message in recent_chat:
            await self._send_message_to_connection(connection_id, message)

    def _add_to_history(self, message: WebSocketMessage):
        """Agregar mensaje al historial"""
        message_type = message.type
        self.message_history[message_type].append(message)
        
        # Mantener límite de historial
        if len(self.message_history[message_type]) > self.max_history_per_type:
            self.message_history[message_type] = self.message_history[message_type][-self.max_history_per_type:]

    async def _cleanup_task(self):
        """Tarea de limpieza periódica"""
        while True:
            try:
                await asyncio.sleep(60)  # Cada minuto
                await self._cleanup_stale_connections()
            except Exception as e:
                logger.error(f"Error en cleanup task: {e}")

    async def _ping_task(self):
        """Tarea de ping periódico"""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                await self._ping_all_connections()
            except Exception as e:
                logger.error(f"Error en ping task: {e}")

    async def _cleanup_stale_connections(self):
        """Limpiar conexiones obsoletas"""
        now = datetime.now()
        stale_connections = []
        
        for connection_id, connection_info in self.active_connections.items():
            time_since_ping = (now - connection_info.last_ping).total_seconds()
            
            if time_since_ping > self.connection_timeout:
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            logger.info(f"Limpiando conexión obsoleta: {connection_id}")
            await self.disconnect(connection_id)

    async def _ping_all_connections(self):
        """Ping a todas las conexiones"""
        ping_message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM,
            content={"type": "ping", "timestamp": datetime.now().isoformat()},
            timestamp=datetime.now().isoformat()
        )
        
        disconnected = []
        for connection_id, connection_info in self.active_connections.items():
            try:
                await self._send_message_to_connection(connection_id, ping_message)
                connection_info.last_ping = datetime.now()
            except:
                disconnected.append(connection_id)
        
        for connection_id in disconnected:
            await self.disconnect(connection_id)

    def get_connection_stats(self) -> Dict:
        """Obtener estadísticas de conexiones"""
        now = datetime.now()
        
        return {
            "active_connections": len(self.active_connections),
            "total_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "metrics": self.metrics.copy(),
            "topics": list(self.subscriptions.keys()),
            "average_connection_time": sum(
                (now - conn.connection_time).total_seconds() 
                for conn in self.active_connections.values()
            ) / len(self.active_connections) if self.active_connections else 0
        }

# Instancia global del manager
ws_manager = AdvancedWebSocketManager()

# =====================
# HANDLERS DE MENSAJES
# =====================

class MessageHandler:
    """Manejador de diferentes tipos de mensajes"""
    
    @staticmethod
    async def handle_chat_message(connection_id: str, content: Dict):
        """Manejar mensaje de chat"""
        user_message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.CHAT,
            content={
                "message": content.get("message", ""),
                "user": ws_manager.active_connections[connection_id].user_id,
                "role": "user"
            },
            timestamp=datetime.now().isoformat(),
            user_id=ws_manager.active_connections[connection_id].user_id
        )
        
        # Broadcast mensaje del usuario
        await ws_manager.broadcast(user_message)
        
        # Simular respuesta de IA (en producción sería llamada a AI service)
        await asyncio.sleep(1)
        
        ai_response = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.CHAT,
            content={
                "message": await MessageHandler._generate_ai_response(content.get("message", "")),
                "user": "IA Assistant",
                "role": "assistant"
            },
            timestamp=datetime.now().isoformat()
        )
        
        await ws_manager.broadcast(ai_response)

    @staticmethod
    async def _generate_ai_response(user_message: str) -> str:
        """Generar respuesta de IA (simulada)"""
        responses = [
            f"Entiendo que mencionas: '{user_message}'. ¿Podrías ser más específico?",
            f"Basándome en tu consulta sobre '{user_message}', te sugiero revisar los documentos relacionados.",
            f"Para '{user_message}', puedo ayudarte con análisis de datos o procesamiento de documentos.",
            f"Interesante punto sobre '{user_message}'. ¿Te gustaría que ejecute algún análisis específico?",
            f"Sobre '{user_message}': Puedo procesar documentos, generar reportes o hacer análisis estadísticos."
        ]
        
        import random
        return random.choice(responses)

    @staticmethod
    async def handle_subscription(connection_id: str, content: Dict):
        """Manejar suscripción a tópico"""
        topic = content.get("topic")
        action = content.get("action", "subscribe")
        
        if action == "subscribe":
            await ws_manager.subscribe(connection_id, topic)
            response = f"Suscrito a {topic}"
        else:
            await ws_manager.unsubscribe(connection_id, topic)
            response = f"Desuscrito de {topic}"
        
        confirmation = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM,
            content={"message": response, "topic": topic},
            timestamp=datetime.now().isoformat()
        )
        
        await ws_manager.send_message(confirmation, [connection_id])

    @staticmethod
    async def handle_agent_command(connection_id: str, content: Dict):
        """Manejar comando del agente"""
        command = content.get("command", "")
        
        # Simular ejecución del comando
        execution_message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.AGENT_ACTION,
            content={
                "status": "executing",
                "command": command,
                "message": f"Ejecutando: {command}"
            },
            timestamp=datetime.now().isoformat()
        )
        
        await ws_manager.publish_to_topic("agent_actions", execution_message)
        
        # Simular tiempo de procesamiento
        await asyncio.sleep(2)
        
        # Resultado
        result_message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.AGENT_ACTION,
            content={
                "status": "completed",
                "command": command,
                "result": f"Comando '{command}' ejecutado exitosamente",
                "details": {
                    "documents_processed": 5,
                    "insights_generated": 3,
                    "time_taken": "2.1s"
                }
            },
            timestamp=datetime.now().isoformat()
        )
        
        await ws_manager.publish_to_topic("agent_actions", result_message)

# =====================
# WEBSOCKET ENDPOINT
# =====================

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint principal de WebSocket"""
    connection_id = None
    
    try:
        # Conectar
        connection_id = await ws_manager.connect(websocket)
        
        while True:
            # Recibir mensaje
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                ws_manager.metrics["messages_received"] += 1
                
                # Procesar según tipo de mensaje
                message_type = message_data.get("type", "chat")
                content = message_data.get("content", {})
                
                if message_type == "chat":
                    await MessageHandler.handle_chat_message(connection_id, content)
                elif message_type == "subscribe" or message_type == "unsubscribe":
                    await MessageHandler.handle_subscription(connection_id, {
                        "topic": content.get("topic"),
                        "action": message_type
                    })
                elif message_type == "agent_command":
                    await MessageHandler.handle_agent_command(connection_id, content)
                elif message_type == "ping":
                    # Actualizar último ping
                    if connection_id in ws_manager.active_connections:
                        ws_manager.active_connections[connection_id].last_ping = datetime.now()
                    
                    # Responder pong
                    pong_message = WebSocketMessage(
                        id=str(uuid.uuid4()),
                        type=MessageType.SYSTEM,
                        content={"type": "pong", "timestamp": datetime.now().isoformat()},
                        timestamp=datetime.now().isoformat()
                    )
                    await ws_manager.send_message(pong_message, [connection_id])
                
            except json.JSONDecodeError:
                error_message = WebSocketMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.ERROR,
                    content={"error": "Formato JSON inválido"},
                    timestamp=datetime.now().isoformat()
                )
                await ws_manager.send_message(error_message, [connection_id])
                ws_manager.metrics["errors"] += 1
            
    except WebSocketDisconnect:
        if connection_id:
            await ws_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"Error en WebSocket {connection_id}: {e}")
        if connection_id:
            await ws_manager.disconnect(connection_id)

# =====================
# RUTAS HTTP PARA WEBSOCKET
# =====================

@router.get("/ws/stats")
async def get_websocket_stats():
    """Obtener estadísticas de WebSocket"""
    return ws_manager.get_connection_stats()

@router.post("/ws/broadcast")
async def broadcast_message(message: Dict):
    """Broadcast mensaje vía HTTP"""
    ws_message = WebSocketMessage(
        id=str(uuid.uuid4()),
        type=MessageType(message.get("type", "notification")),
        content=message.get("content", {}),
        timestamp=datetime.now().isoformat()
    )
    
    await ws_manager.broadcast(ws_message)
    return {"status": "sent", "connections": len(ws_manager.active_connections)}

@router.post("/ws/publish/{topic}")
async def publish_to_topic(topic: str, message: Dict):
    """Publicar mensaje a tópico específico"""
    ws_message = WebSocketMessage(
        id=str(uuid.uuid4()),
        type=MessageType(message.get("type", "notification")),
        content=message.get("content", {}),
        timestamp=datetime.now().isoformat()
    )
    
    await ws_manager.publish_to_topic(topic, ws_message)
    
    subscribers = len(ws_manager.subscriptions.get(topic, set()))
    return {"status": "published", "topic": topic, "subscribers": subscribers}

# =====================
# NOTIFICACIONES DEL SISTEMA
# =====================

async def send_system_notification(message: str, notification_type: str = "info"):
    """Enviar notificación del sistema"""
    notification = WebSocketMessage(
        id=str(uuid.uuid4()),
        type=MessageType.NOTIFICATION,
        content={
            "message": message,
            "type": notification_type,
            "timestamp": datetime.now().isoformat()
        },
        timestamp=datetime.now().isoformat()
    )
    
    await ws_manager.broadcast(notification)

async def send_analytics_update(analysis_id: str, progress: float, status: str):
    """Enviar actualización de análisis"""
    update = WebSocketMessage(
        id=str(uuid.uuid4()),
        type=MessageType.ANALYTICS_UPDATE,
        content={
            "analysis_id": analysis_id,
            "progress": progress,
            "status": status,
            "timestamp": datetime.now().isoformat()
        },
        timestamp=datetime.now().isoformat()
    )
    
    await ws_manager.publish_to_topic("analytics", update)

async def send_training_update(project_id: str, epoch: int, loss: float, accuracy: float):
    """Enviar actualización de entrenamiento"""
    update = WebSocketMessage(
        id=str(uuid.uuid4()),
        type=MessageType.TRAINING_UPDATE,
        content={
            "project_id": project_id,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        },
        timestamp=datetime.now().isoformat()
    )
    
    await ws_manager.publish_to_topic("training", update)