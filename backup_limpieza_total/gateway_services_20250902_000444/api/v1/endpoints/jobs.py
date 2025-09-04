"""
Jobs API endpoints for managing background tasks.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import logging

from ....jobs.manager import job_manager
from ....jobs.models import JobInDB, JobCreate, JobUpdate, JobStatus, JobType
from ....ws.manager import websocket_manager

router = APIRouter()
logger = logging.getLogger(__name__)

class JobResponse(BaseModel):
    ok: bool = True
    data: JobInDB

class JobListResponse(BaseModel):
    ok: bool = True
    data: List[JobInDB]
    total: int

@router.post("/", response_model=JobResponse, status_code=201)
async def create_job(job_create: JobCreate):
    """
    Create a new background job.
    
    - **job_type**: Type of job to create (report_generation, document_processing, etc.)
    - **params**: Parameters for the job
    - **created_by**: ID of the user creating the job
    - **project_id**: Optional project ID this job is associated with
    """
    try:
        # Create a new job in the database
        job = JobInDB(
            **job_create.dict(),
            status=JobStatus.PENDING,
            progress=0
        )
        
        # Submit the job for execution
        created_job = await job_manager.submit_job(job)
        
        return {"ok": True, "data": created_job}
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get the status of a specific job by ID.
    """
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"ok": True, "data": job}

@router.get("/", response_model=JobListResponse)
async def list_jobs(
    status: Optional[JobStatus] = None,
    job_type: Optional[JobType] = None,
    project_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List all jobs with optional filtering.
    
    - **status**: Filter by job status
    - **job_type**: Filter by job type
    - **project_id**: Filter by project ID
    - **limit**: Maximum number of jobs to return
    - **offset**: Number of jobs to skip
    """
    try:
        jobs = await job_manager.list_jobs(
            status=status,
            job_type=job_type,
            project_id=project_id,
            limit=limit,
            offset=offset
        )
        return {
            "ok": True,
            "data": jobs,
            "total": len(jobs)  # Note: For pagination, you might want to get total count from DB
        }
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str):
    """
    Cancel a running job.
    """
    try:
        cancelled = await job_manager.cancel_job(job_id)
        if not cancelled:
            raise HTTPException(status_code=404, detail="Job not found or not running")
        
        job = await job_manager.get_job(job_id)
        return {"ok": True, "data": job}
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job updates.
    """
    # Accept the WebSocket connection
    await websocket.accept()
    
    # Check if job exists
    job = await job_manager.get_job(job_id)
    if not job:
        await websocket.close(code=1008, reason="Job not found")
        return
    
    # Subscribe to job updates
    client_id = f"job_{job_id}_{id(websocket)}"
    await websocket_manager.connect(websocket, client_id)
    await websocket_manager.subscribe(client_id, f"job_updates:{job_id}")
    
    # Send current job status
    await websocket.send_json({
        "ok": True,
        "data": {
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress,
            "message": job.logs[-1] if job.logs else ""
        }
    })
    
    try:
        # Keep the connection open
        while True:
            # Wait for a message (we don't expect any, but this keeps the connection alive)
            data = await websocket.receive_text()
            
            # If we receive a ping, respond with pong
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        # Clean up on disconnect
        await websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {str(e)}", exc_info=True)
        await websocket_manager.disconnect(client_id)
        await websocket.close(code=1011, reason=str(e))
