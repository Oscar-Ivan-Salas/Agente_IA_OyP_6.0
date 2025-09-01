"""
Gestor de trabajos para el sistema de orquestación.

Este módulo proporciona la funcionalidad para crear, ejecutar y hacer seguimiento
de trabajos asíncronos en el sistema.
"""
import asyncio
import logging
from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime, timedelta

from .models import Job, JobStatus, JobType, JobCreate, JobUpdate

logger = logging.getLogger(__name__)

class JobManager:
    """
    Gestiona la creación, ejecución y seguimiento de trabajos asíncronos.
    """
    
    def __init__(self):
        """Inicializa el gestor de trabajos."""
        self._jobs: Dict[UUID, Job] = {}
        self._tasks: Dict[UUID, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        
        # Configurar limpieza periódica de trabajos antiguos
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def create_job(self, job_create: JobCreate) -> Job:
        """
        Crea un nuevo trabajo.
        
        Args:
            job_create: Datos para crear el trabajo.
            
        Returns:
            El trabajo creado.
        """
        job = Job(
            type=job_create.type,
            metadata=job_create.metadata
        )
        
        async with self._lock:
            self._jobs[job.id] = job
            
        logger.info(f"Trabajo creado: {job.id} ({job.type})")
        return job
    
    async def get_job(self, job_id: UUID) -> Optional[Job]:
        """
        Obtiene un trabajo por su ID.
        
        Args:
            job_id: ID del trabajo a obtener.
            
        Returns:
            El trabajo si existe, None en caso contrario.
        """
        return self._jobs.get(job_id)
    
    async def list_jobs(
        self, 
        status: Optional[JobStatus] = None, 
        job_type: Optional[JobType] = None,
        limit: int = 100
    ) -> List[Job]:
        """
        Lista los trabajos que coincidan con los filtros.
        
        Args:
            status: Filtrar por estado.
            job_type: Filtrar por tipo de trabajo.
            limit: Número máximo de trabajos a devolver.
            
        Returns:
            Lista de trabajos que coinciden con los filtros.
        """
        filtered = list(self._jobs.values())
        
        if status is not None:
            filtered = [j for j in filtered if j.status == status]
            
        if job_type is not None:
            filtered = [j for j in filtered if j.type == job_type]
        
        # Ordenar por fecha de creación (más recientes primero)
        filtered.sort(key=lambda j: j.created_at, reverse=True)
        
        return filtered[:limit]
    
    async def update_job(
        self, 
        job_id: UUID, 
        update: JobUpdate,
        partial: bool = True
    ) -> Optional[Job]:
        """
        Actualiza un trabajo existente.
        
        Args:
            job_id: ID del trabajo a actualizar.
            update: Datos de actualización.
            partial: Si es True, solo actualiza los campos proporcionados.
            
        Returns:
            El trabajo actualizado, o None si no existe.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
                
            # Actualizar campos proporcionados
            if update.status is not None:
                job.status = update.status
                
            if update.progress is not None:
                job.progress = update.progress
                
            if update.result is not None:
                job.result = update.result
                
            if update.error is not None:
                job.error = update.error
            
            # Actualizar marcas de tiempo según el estado
            if job.status == JobStatus.RUNNING and job.started_at is None:
                job.started_at = datetime.utcnow()
            elif job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at is None:
                    job.completed_at = datetime.utcnow()
        
        return job
    
    async def _periodic_cleanup(self, max_age_hours: int = 24):
        """
        Limpia periódicamente los trabajos antiguos.
        
        Args:
            max_age_hours: Edad máxima en horas para conservar los trabajos.
        """
        while True:
            try:
                await asyncio.sleep(3600)  # Ejecutar cada hora
                
                cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
                to_remove = []
                
                async with self._lock:
                    for job_id, job in list(self._jobs.items()):
                        if (
                            job.completed_at and job.completed_at < cutoff
                        ) or (
                            job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED) 
                            and job.completed_at is None 
                            and job.created_at < cutoff - timedelta(hours=1)
                        ):
                            to_remove.append(job_id)
                    
                    for job_id in to_remove:
                        self._jobs.pop(job_id, None)
                        self._tasks.pop(job_id, None)
                
                if to_remove:
                    logger.info(f"Limpieza de trabajos: eliminados {len(to_remove)} trabajos antiguos")
                    
            except Exception as e:
                logger.error(f"Error en la limpieza periódica de trabajos: {e}", exc_info=True)
    
    def __del__(self):
        """Asegura que las tareas en segundo plano se cancelen al destruir el gestor."""
        if hasattr(self, '_cleanup_task') and self._cleanup_task:
            self._cleanup_task.cancel()
