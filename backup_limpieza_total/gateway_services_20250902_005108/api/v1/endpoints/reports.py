"""
Reports API endpoints.
"""
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/weekly/{project_id}")
async def generate_weekly_report(project_id: str):
    """Generate a weekly report for a project."""
    return {"ok": True, "data": {"url": f"/reports/weekly_{project_id}.pdf"}}

@router.get("/final/{project_id}")
async def generate_final_report(project_id: str):
    """Generate a final report for a project."""
    return {"ok": True, "data": {"url": f"/reports/final_{project_id}.pdf"}}

@router.get("/gantt/{project_id}")
async def generate_gantt(project_id: str):
    """Generate a Gantt chart for a project."""
    return {"ok": True, "data": {"url": f"/reports/gantt_{project_id}.png"}}
