"""
Módulo de orquestación de trabajos.

Este paquete maneja la creación, ejecución y seguimiento de trabajos asíncronos
dentro del sistema.
"""
from .manager import JobManager
from .models import Job, JobStatus, JobType

# Exportar la instancia del gestor de trabajos
job_manager = JobManager()

__all__ = ["job_manager", "Job", "JobStatus", "JobType"]
