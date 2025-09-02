"""
Módulo de rutas para el Gateway

Este paquete contiene todos los routers de la aplicación.
"""

# Importar routers para que estén disponibles al importar el paquete
from . import api
from . import dashboard
from . import services
from . import websocket

__all__ = ['api', 'dashboard', 'services', 'websocket']