#!/usr/bin/env python3
"""
🚀 AGENTE IA OYP 6.0 - DOCUMENT PROCESSOR
==========================================
Microservicio para gestión, extracción y procesamiento de documentos.
Archivo: services/document-processor/app.py
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

# FastAPI imports
import aiofiles
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Dependencias de procesamiento
import pypdf

# =====================
# CONFIGURACIÓN GLOBAL
# =====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directorios de datos
DATA_DIR = Path(os.getenv("DATA_DIR", "../../data")).resolve()
PROJECTS_DIR = DATA_DIR / "projects"

# URL del servicio de IA
AI_ENGINE_URL = "http://localhost:8001"

# Simulación de base de datos de trabajos en segundo plano
JOBS_DB: Dict[str, Dict[str, Any]] = {}

# =====================
# APLICACIÓN FASTAPI
# =====================

app = FastAPI(
    title="📄 Document Processor Service",
    description="Microservicio para pipeline de procesamiento de documentos.",
    version="1.0.0"
)

# =====================
# MODELOS PYDANTIC
# =====================

class ProjectCreateRequest(BaseModel):
    name: Optional[str] = None

class PipelinePlanRequest(BaseModel):
    sample_files: List[str]
    goals: List[str] = Field(..., description="Ej: ['normalize', 'redact', 'summarize']")

class PipelineRunRequest(BaseModel):
    steps: List[Dict[str, Any]]

# =====================
# UTILIDADES Y HELPERS
# =====================

def build_meta(start_time: float) -> Dict[str, Any]:
    return {"took_ms": int((time.time() - start_time) * 1000), "ts": datetime.now(timezone.utc).isoformat()}

def create_response(data: Any, ok: bool = True, start_time: float = time.time()) -> JSONResponse:
    content = {"ok": ok, "data": data, "meta": build_meta(start_time)}
    return JSONResponse(content=content)

def create_error_response(code: str, message: str, status_code: int = 500, start_time: float = time.time()) -> JSONResponse:
    content = {"ok": False, "error": {"code": code, "message": message}, "meta": build_meta(start_time)}
    return JSONResponse(status_code=status_code, content=content)

def get_project_path(project_id: str) -> Optional[Path]:
    project_path = PROJECTS_DIR / project_id
    return project_path if project_path.exists() and project_path.is_dir() else None

def generate_checksum(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# =====================
# LÓGICA DE PIPELINE
# =====================

async def run_pipeline_background(project_id: str, steps: List[Dict], job_id: str):
    """Función que se ejecuta en segundo plano para procesar el pipeline."""
    JOBS_DB[job_id]["status"] = "running"
    logger.info(f"Iniciando pipeline para proyecto {project_id} (Job: {job_id})")
    
    project_path = get_project_path(project_id)
    raw_path = project_path / "raw"
    clean_path = project_path / "clean"
    json_path = project_path / "json"
    clean_path.mkdir(exist_ok=True)
    json_path.mkdir(exist_ok=True)

    files_to_process = list(raw_path.glob("*"))
    total_files = len(files_to_process)

    for i, file_path in enumerate(files_to_process):
        JOBS_DB[job_id]["progress"] = (i + 1) / total_files
        try:
            # 1. Extraer texto
            text_content = ""
            if file_path.suffix.lower() == ".pdf":
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        text_content += page.extract_text() + "\n"
            else: # Asumir texto plano
                async with aiofiles.open(file_path, 'r') as f:
                    text_content = await f.read()

            # 2. Limpieza básica (como se describe en el prompt)
            clean_text = text_content.replace('\u0000', '').strip()
            
            # Guardar texto limpio
            async with aiofiles.open(clean_path / f"{file_path.stem}.txt", 'w') as f:
                await f.write(clean_text)

            # 3. Crear JSON normalizado (simplificado)
            normalized_json = {
                "doc_id": file_path.stem,
                "project_id": project_id,
                "content": {"text": clean_text},
                "audit": {"pipeline": steps}
            }
            async with aiofiles.open(json_path / f"{file_path.stem}.json", 'w') as f:
                await f.write(json.dumps(normalized_json, indent=2))

        except Exception as e:
            logger.error(f"Error procesando {file_path.name} en job {job_id}: {e}")
            JOBS_DB[job_id]["errors"].append(f"{file_path.name}: {e}")

    JOBS_DB[job_id]["status"] = "completed"
    JOBS_DB[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(f"Pipeline completado para proyecto {project_id} (Job: {job_id})")

# =====================
# ENDPOINTS DE LA API
# =====================

@app.on_event("startup")
async def startup_event():
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de datos: {PROJECTS_DIR}")

@app.get("/health")
async def health_check():
    return create_response({"status": "healthy"})

@app.post("/projects")
async def create_project(req: ProjectCreateRequest):
    start_time = time.time()
    project_id = req.name or f"proj_{int(time.time())}"
    project_path = PROJECTS_DIR / project_id
    
    if project_path.exists():
        return create_error_response("PROJECT_EXISTS", f"El proyecto '{project_id}' ya existe.", 409, start_time)
    
    # Crear toda la estructura de directorios
    for sub_dir in ["raw", "clean", "json", "logs", "manifests", "presets"]:
        (project_path / sub_dir).mkdir(parents=True, exist_ok=True)
        
    return create_response({"project_id": project_id}, start_time=start_time)

@app.get("/projects")
async def list_projects():
    start_time = time.time()
    projects = []
    for p_path in PROJECTS_DIR.iterdir():
        if p_path.is_dir():
            raw_count = len(list((p_path / "raw").glob("*")))
            projects.append({"id": p_path.name, "raw_files": raw_count})
    return create_response(projects, start_time=start_time)

@app.post("/projects/{project_id}/upload")
async def upload_files(project_id: str, files: List[UploadFile] = File(...)):
    start_time = time.time()
    project_path = get_project_path(project_id)
    if not project_path:
        return create_error_response("NOT_FOUND", f"Proyecto '{project_id}' no encontrado.", 404, start_time)

    manifests_path = project_path / "manifests"
    raw_path = project_path / "raw"
    uploaded_files_info = []

    for file in files:
        file_location = raw_path / file.filename
        async with aiofiles.open(file_location, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        manifest = {
            "filename": file.filename,
            "size": file_location.stat().st_size,
            "checksum": generate_checksum(file_location),
            "content_type": file.content_type,
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
        async with aiofiles.open(manifests_path / f"{file.filename}.json", 'w') as mf:
            await mf.write(json.dumps(manifest, indent=2))
        
        uploaded_files_info.append(manifest)

    return create_response({"uploaded_files": uploaded_files_info}, start_time=start_time)

@app.get("/projects/{project_id}/files")
async def list_project_files(project_id: str, stage: str = "raw"):
    start_time = time.time()
    project_path = get_project_path(project_id)
    if not project_path:
        return create_error_response("NOT_FOUND", f"Proyecto '{project_id}' no encontrado.", 404, start_time)
    
    stage_path = project_path / stage
    if not stage_path.exists():
        return create_error_response("BAD_REQUEST", f"La etapa '{stage}' no es válida.", 400, start_time)
        
    files = [f.name for f in stage_path.iterdir()]
    return create_response({"files": files, "stage": stage, "count": len(files)}, start_time=start_time)

@app.post("/projects/{project_id}/pipeline/plan")
async def plan_pipeline(project_id: str, req: PipelinePlanRequest):
    start_time = time.time()
    # Llama a 8001/analyze_text para sugerir pasos (simulado)
    try:
        async with httpx.AsyncClient(base_url=AI_ENGINE_URL, timeout=10.0) as client:
            # En un caso real, enviaríamos más contexto
            response = await client.post("/analyze_text", json={
                "text": f"Plan pipeline for project {project_id} with goals: {req.goals}"
            })
            response.raise_for_status()
            ai_data = response.json()
            
            # Simular una traducción de la respuesta de la IA a un plan
            plan = {
                "suggested_steps": [
                    {"step": "extract_text", "params": {"ocr_if_needed": True}},
                    {"step": "clean_text", "params": {"mode": "NFKC"}},
                    {"step": "summarize", "params": {"model": "llama"}}
                ],
                "ai_analysis": ai_data.get("data", {})
            }
            return create_response(plan, start_time=start_time)
    except Exception as e:
        return create_error_response("AI_SERVICE_ERROR", f"No se pudo contactar al AI Engine: {e}", 503, start_time)

@app.post("/projects/{project_id}/pipeline/run")
async def run_pipeline(project_id: str, req: PipelineRunRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    project_path = get_project_path(project_id)
    if not project_path:
        return create_error_response("NOT_FOUND", f"Proyecto '{project_id}' no encontrado.", 404, start_time)

    job_id = f"job_{project_id}_{int(time.time())}"
    JOBS_DB[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "errors": []
    }
    
    background_tasks.add_task(run_pipeline_background, project_id, req.steps, job_id)
    
    return create_response({"job_id": job_id, "status": "queued"}, start_time=start_time)

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    start_time = time.time()
    job = JOBS_DB.get(job_id)
    if not job:
        return create_error_response("NOT_FOUND", f"Job '{job_id}' no encontrado.", 404, start_time)
    return create_response(job, start_time=start_time)

@app.get("/projects/{project_id}/export")
async def export_project(project_id: str, format: str = "zip-json"):
    start_time = time.time()
    project_path = get_project_path(project_id)
    if not project_path:
        return create_error_response("NOT_FOUND", f"Proyecto '{project_id}' no encontrado.", 404, start_time)

    if format != "zip-json":
        return create_error_response("BAD_REQUEST", "Formato de exportación no soportado.", 400, start_time)

    json_path = project_path / "json"
    zip_path_base = project_path / f"export_{project_id}"
    
    shutil.make_archive(str(zip_path_base), 'zip', str(json_path))
    
    zip_path = Path(f"{zip_path_base}.zip")
    if zip_path.exists():
        return FileResponse(zip_path, media_type='application/zip', filename=zip_path.name)
    else:
        return create_error_response("EXPORT_FAILED", "No se pudo crear el archivo ZIP.", 500, start_time)

# =====================
# MAIN
# =====================

if __name__ == "__main__":
    logger.info("🚀 Iniciando Document Processor Service...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
