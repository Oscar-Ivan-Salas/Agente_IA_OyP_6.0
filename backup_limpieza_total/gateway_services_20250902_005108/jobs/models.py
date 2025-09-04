"""
Job models and database operations.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import uuid
import json

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobType(str, Enum):
    REPORT_GENERATION = "report_generation"
    DOCUMENT_PROCESSING = "document_processing"
    DATA_ANALYSIS = "data_analysis"
    AI_PROCESSING = "ai_processing"
    OTHER = "other"

class JobBase(BaseModel):
    """Base job model."""
    job_type: JobType = Field(..., description="Type of the job")
    params: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")
    created_by: str = Field(..., description="ID of the user who created the job")
    project_id: Optional[str] = Field(None, description="Related project ID")

class JobCreate(JobBase):
    """Model for creating a new job."""
    pass

class JobUpdate(BaseModel):
    """Model for updating a job."""
    status: Optional[JobStatus] = None
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage (0-100)")
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: Optional[List[str]] = None

class JobInDB(JobBase):
    """Job model as stored in the database."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique job ID")
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = Field(default_factory=list)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "job_type": "report_generation",
                "status": "completed",
                "progress": 100,
                "params": {"report_type": "weekly"},
                "created_by": "user123",
                "project_id": "project456",
                "created_at": "2023-06-01T10:00:00Z",
                "updated_at": "2023-06-01T10:05:00Z",
                "started_at": "2023-06-01T10:00:30Z",
                "completed_at": "2023-06-01T10:05:00Z",
                "result": {"report_url": "/reports/weekly_20230601.pdf"},
                "logs": ["Report generation started", "Data fetched", "Report generated"]
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        data = self.dict()
        # Convert enums to strings
        data["status"] = self.status.value
        data["job_type"] = self.job_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobInDB':
        """Create model from dictionary."""
        # Convert string enums back to enum values
        if "status" in data and isinstance(data["status"], str):
            data["status"] = JobStatus(data["status"])
        if "job_type" in data and isinstance(data["job_type"], str):
            data["job_type"] = JobType(data["job_type"])
        return cls(**data)
