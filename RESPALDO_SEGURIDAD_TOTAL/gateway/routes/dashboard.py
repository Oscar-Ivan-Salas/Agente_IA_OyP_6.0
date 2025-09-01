"""
📱 RUTAS DEL DASHBOARD
======================
Rutas para servir el dashboard y sus estadísticas
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import random
import logging

# Configuración de logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"],
    responses={404: {"description": "No encontrado"}},
)

# Datos de ejemplo para estadísticas
def generate_sample_stats():
    now = datetime.utcnow()
    return {
        "system_status": {
            "status": "operational",
            "last_updated": now.isoformat(),
            "services": {
                "ai_engine": {
                    "status": "operational",
                    "response_time": random.uniform(50, 200)
                },
                "document_processor": {
                    "status": "operational",
                    "response_time": random.uniform(30, 150)
                },
                "analytics_engine": {
                    "status": "operational",
                    "response_time": random.uniform(40, 180)
                },
                "report_generator": {
                    "status": "operational",
                    "response_time": random.uniform(60, 250)
                }
            }
        },
        "usage_metrics": {
            "total_requests": random.randint(1000, 5000),
            "success_rate": round(random.uniform(95, 99.9), 2),
            "active_users": random.randint(5, 50),
            "data_processed_mb": round(random.uniform(100, 5000), 2)
        },
        "recent_activity": [
            {
                "id": i,
                "service": random.choice(["ai_engine", "document_processor", "analytics_engine", "report_generator"]),
                "action": random.choice(["process", "analyze", "generate", "train"]),
                "timestamp": (now - timedelta(minutes=random.randint(1, 60))).isoformat(),
                "status": random.choice(["completed", "in_progress", "failed"])
            } for i in range(5)
        ]
    }

@router.get("/stats", response_model=dict)
async def get_dashboard_stats():
    """
    Obtiene las estadísticas del dashboard
    """
    try:
        stats = generate_sample_stats()
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": stats,
                "message": "Estadísticas obtenidas correctamente"
            }
        )
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Error interno al obtener estadísticas"}
        )
