"""
Tasks API endpoints.
"""
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/{task_id}")
async def get_task(task_id: str):
    """Get a specific task by ID."""
    return {"ok": True, "data": {"id": task_id}}

@router.put("/{task_id}")
async def update_task(task_id: str):
    """Update a task."""
    return {"ok": True, "data": {"id": task_id, "updated": True}}

@router.post("/import")
async def import_tasks():
    """Import tasks from CSV/Excel."""
    return {"ok": True, "data": {"imported": 0}}

@router.get("/project/{project_id}")
async def list_project_tasks(project_id: str):
    """List all tasks for a project."""
    return {"ok": True, "data": [], "total": 0}
