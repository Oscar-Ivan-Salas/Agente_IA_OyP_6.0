"""
Gateway Service - Main Entry Point

This module initializes and runs the FastAPI application for the Gateway service.
"""
import uvicorn
import logging
from fastapi import FastAPI, Request, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
import os

from .core.config import settings
from .api.v1.api import api_router
from .ws.manager import websocket_manager
from .jobs.manager import job_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Gateway service for the Agente IA OYP 6.0 platform",
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url=None
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = os.path.abspath("static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include API routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"ok": False, "error": {"code": "VALIDATION_ERROR", "msg": str(exc)}},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle global exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"ok": False, "error": {"code": "INTERNAL_SERVER_ERROR", "msg": "An unexpected error occurred"}},
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "ok": True,
        "status": "healthy",
        "service": "gateway",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }

# Serve dashboard
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard."""
    if not os.path.exists(INDEX_PATH):
        return JSONResponse(
            {"ok": False, "error": {"code": "NO_INDEX", "msg": "Copia index.html a static/index.html"}}, 
            status_code=500
        )
    return FileResponse(INDEX_PATH)

# WebSocket endpoint for job updates
@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
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

# WebSocket endpoint for chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for chat functionality."""
    await websocket.accept()
    client_id = f"chat_{id(websocket)}"
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get('type') == 'chat_message':
                # Echo the message back to the client
                await websocket.send_json({
                    'type': 'chat_message',
                    'sender': 'Sistema',
                    'message': f'Recibido: {data.get("message", "")}',
                    'timestamp': data.get('timestamp')
                })
    except Exception as e:
        logger.error(f"Chat WebSocket error: {str(e)}")
    finally:
        await websocket.close()

# Register job handlers
# Example:
# @job_manager.register_job_handler("example_job_type")
# async def handle_example_job(job: JobInDB):
#     """Example job handler."""
#     # Job processing logic here
#     return {"result": "success"}

def start():
    """Start the Gateway service."""
    uvicorn.run(
        "services.gateway.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )

if __name__ == "__main__":
    start()
