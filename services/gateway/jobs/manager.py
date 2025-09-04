"""
Job manager for handling background tasks and job orchestration.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Coroutine
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session as DBSession
import uuid
import os

from ..core.config import settings
from .models import JobInDB, JobStatus, JobUpdate, JobType
from ..ws.manager import websocket_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy models
Base = declarative_base()

class DBJob(Base):
    """Database model for jobs."""
    __tablename__ = "jobs"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    job_type = Column(String(50), nullable=False)
    status = Column(String(20), default=JobStatus.PENDING.value, nullable=False)
    progress = Column(Integer, default=0)
    params = Column(Text, default="{}")  # JSON string
    result = Column(Text, default=None)  # JSON string
    error = Column(Text, default=None)
    logs = Column(Text, default="[]")  # JSON array of strings
    created_by = Column(String(100), nullable=False)
    project_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    def to_model(self) -> JobInDB:
        """Convert to Pydantic model."""
        return JobInDB(
            id=self.id,
            job_type=self.job_type,
            status=JobStatus(self.status) if self.status else JobStatus.PENDING,
            progress=self.progress or 0,
            params=json.loads(self.params) if self.params else {},
            result=json.loads(self.result) if self.result else None,
            error=self.error,
            logs=json.loads(self.logs) if self.logs else [],
            created_by=self.created_by,
            project_id=self.project_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            started_at=self.started_at,
            completed_at=self.completed_at
        )

class JobManager:
    """Manages background jobs and their lifecycle."""
    
    def __init__(self, db_url: str = None):
        """Initialize the job manager with a database connection."""
        self.db_url = db_url or settings.DATABASE_URL
        self.engine = create_engine(
            self.db_url, connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._running_jobs: Dict[str, asyncio.Task] = {}
        self._job_handlers: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_old_jobs())
    
    def get_db(self) -> DBSession:
        """Get a database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    async def create_job(self, job_create: JobInDB) -> JobInDB:
        """Create a new job in the database."""
        db = next(self.get_db())
        try:
            db_job = DBJob(
                id=job_create.id,
                job_type=job_create.job_type.value,
                status=job_create.status.value,
                progress=job_create.progress,
                params=json.dumps(job_create.params),
                created_by=job_create.created_by,
                project_id=job_create.project_id,
                created_at=job_create.created_at,
                updated_at=job_create.updated_at,
                started_at=job_create.started_at,
                completed_at=job_create.completed_at,
                logs=json.dumps(job_create.logs or [])
            )
            db.add(db_job)
            db.commit()
            db.refresh(db_job)
            return db_job.to_model()
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating job: {str(e)}")
            raise
        finally:
            db.close()
    
    async def get_job(self, job_id: str) -> Optional[JobInDB]:
        """Get a job by ID."""
        db = next(self.get_db())
        try:
            db_job = db.query(DBJob).filter(DBJob.id == job_id).first()
            return db_job.to_model() if db_job else None
        finally:
            db.close()
    
    async def update_job(
        self, 
        job_id: str, 
        job_update: JobUpdate
    ) -> Optional[JobInDB]:
        """Update a job's status and other fields."""
        db = next(self.get_db())
        try:
            db_job = db.query(DBJob).filter(DBJob.id == job_id).first()
            if not db_job:
                return None
                
            # Update fields if they are provided
            if job_update.status:
                db_job.status = job_update.status.value
                
                # Update timestamps based on status
                if job_update.status == JobStatus.RUNNING and not db_job.started_at:
                    db_job.started_at = datetime.utcnow()
                elif job_update.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    db_job.completed_at = datetime.utcnow()
            
            if job_update.progress is not None:
                db_job.progress = job_update.progress
                
            if job_update.result is not None:
                db_job.result = json.dumps(job_update.result)
                
            if job_update.error is not None:
                db_job.error = job_update.error
                
            if job_update.logs is not None:
                current_logs = json.loads(db_job.logs) if db_job.logs else []
                current_logs.extend(job_update.logs)
                db_job.logs = json.dumps(current_logs)
            
            db_job.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(db_job)
            
            # Notify WebSocket subscribers
            await self._notify_job_update(db_job.to_model())
            
            return db_job.to_model()
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating job {job_id}: {str(e)}")
            raise
        finally:
            db.close()
    
    async def _notify_job_update(self, job: JobInDB):
        """Send WebSocket notification about job update."""
        await websocket_manager.broadcast(
            {
                "job_id": job.id,
                "status": job.status,
                "progress": job.progress,
                "message": job.logs[-1] if job.logs else ""
            },
            f"job_updates:{job.id}"
        )
    
    async def _cleanup_old_jobs(self):
        """Periodically clean up old completed/failed jobs."""
        while True:
            try:
                db = next(self.get_db())
                try:
                    # Delete jobs older than 30 days
                    cutoff = datetime.utcnow() - timedelta(days=30)
                    db.query(DBJob).filter(
                        DBJob.status.in_([JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]),
                        DBJob.updated_at < cutoff
                    ).delete()
                    db.commit()
                    logger.info("Cleaned up old jobs")
                except Exception as e:
                    logger.error(f"Error cleaning up jobs: {str(e)}")
                    db.rollback()
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
            
            # Run cleanup every hour
            await asyncio.sleep(3600)
    
    async def register_job_handler(self, job_type: str, handler: Callable[..., Coroutine[Any, Any, Any]]):
        """Register a handler function for a specific job type."""
        self._job_handlers[job_type] = handler
    
    async def submit_job(self, job_create: JobInDB) -> JobInDB:
        """Submit a new job for execution."""
        # Create the job in the database
        job = await self.create_job(job_create)
        
        # Start the job in the background
        self._running_jobs[job.id] = asyncio.create_task(
            self._execute_job(job)
        )
        
        return job
    
    async def _execute_job(self, job: JobInDB):
        """Execute a job with the appropriate handler."""
        job_id = job.id
        
        try:
            # Update status to running
            await self.update_job(job_id, JobUpdate(
                status=JobStatus.RUNNING,
                progress=0,
                logs=[f"Starting job: {job.job_type}"]
            ))
            
            # Get the appropriate handler
            handler = self._job_handlers.get(job.job_type.value)
            if not handler:
                raise ValueError(f"No handler registered for job type: {job.job_type}")
            
            # Execute the handler
            result = await handler(job)
            
            # Update job status to completed
            await self.update_job(job_id, JobUpdate(
                status=JobStatus.COMPLETED,
                progress=100,
                result=result,
                logs=["Job completed successfully"]
            ))
            
        except Exception as e:
            # Update job status to failed
            error_msg = str(e)
            logger.error(f"Job {job_id} failed: {error_msg}", exc_info=True)
            await self.update_job(job_id, JobUpdate(
                status=JobStatus.FAILED,
                error=error_msg,
                logs=[f"Job failed: {error_msg}"]
            ))
        finally:
            # Clean up the running job
            self._running_jobs.pop(job_id, None)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id in self._running_jobs:
            # Cancel the background task
            self._running_jobs[job_id].cancel()
            
            # Update job status
            await self.update_job(job_id, JobUpdate(
                status=JobStatus.CANCELLED,
                logs=["Job was cancelled by user"]
            ))
            
            # Clean up
            self._running_jobs.pop(job_id, None)
            return True
        return False
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a job."""
        job = await self.get_job(job_id)
        if not job:
            return None
            
        return {
            "id": job.id,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error": job.error,
            "logs": job.logs[-10:]  # Last 10 log entries
        }
    
    async def list_jobs(
        self, 
        status: Optional[JobStatus] = None, 
        job_type: Optional[JobType] = None,
        project_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filtering."""
        db = next(self.get_db())
        try:
            query = db.query(DBJob)
            
            if status:
                query = query.filter(DBJob.status == status.value)
            if job_type:
                query = query.filter(DBJob.job_type == job_type.value)
            if project_id:
                query = query.filter(DBJob.project_id == project_id)
                
            jobs = query.order_by(DBJob.created_at.desc()).offset(offset).limit(limit).all()
            return [job.to_model().dict() for job in jobs]
        finally:
            db.close()

# Global job manager instance
job_manager = JobManager()
