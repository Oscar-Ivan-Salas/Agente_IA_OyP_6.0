"""
Routers de la API del Gateway.

Este paquete contiene todos los routers que definen los endpoints de la API.
"""

# Importar todos los routers aquí
from . import projects, tasks, daily_logs, risks, reports, jobs, websockets

__all__ = ["projects", "tasks", "daily_logs", "risks", "reports", "jobs", "websockets"]
