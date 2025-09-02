"""
Routers de la API del Gateway.

Este paquete contiene todos los routers que definen los endpoints de la API.
"""

# No importar submódulos aquí para evitar ciclos al importar `gateway.routers`
__all__ = ["projects", "tasks", "daily_logs", "risks", "reports", "websockets"]
