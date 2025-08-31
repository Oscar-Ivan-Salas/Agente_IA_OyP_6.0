"""
Módulo de WebSockets para comunicación en tiempo real.

Este paquete maneja las conexiones WebSocket para actualizaciones en tiempo real
del estado de los trabajos y notificaciones del sistema.
"""
from .manager import ConnectionManager

# Exportar el gestor de conexiones
connection_manager = ConnectionManager()

__all__ = ["connection_manager"]
