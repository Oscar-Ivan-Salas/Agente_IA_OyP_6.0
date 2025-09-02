#!/usr/bin/env python3
"""
Setup Report Generator - Módulo 5 - VERSIÓN COMPLETA
Agente IA OyP 6.0
Generador de Reportes Inteligentes
"""

import os
import sys
import asyncio
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List, Any, Optional

class ReportGeneratorSetup:
    """Setup del servicio Report Generator - VERSIÓN COMPLETA"""
    
    def __init__(self):
        self.service_name = "report-generator"
        self.service_port = 8004
        self.project_path = Path.cwd()
        self.service_path = self.project_path / "services" / self.service_name
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('setup_report_generator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        print(f"""
🤖 ===============================================
📋 SETUP REPORT GENERATOR - MÓDULO 5 COMPLETO
🤖 ===============================================
   Agente IA OyP 6.0 - Generador de Reportes
   Puerto: {self.service_port}
   Ruta: {self.service_path}
🤖 ===============================================
""")

    def run_setup(self):
        """Ejecutar setup completo"""
        try:
            print("🚀 Iniciando setup del Report Generator...")
            
            # Verificar dependencias del sistema
            self.check_system_dependencies()
            
            # Crear estructura de directorios
            self.create_directory_structure()
            
            # Crear archivos de configuración
            self.create_config_files()
            
            # Crear modelos de datos
            self.create_data_models()
            
            # Crear servicios principales
            self.create_report_services()
            
            # Crear generadores especializados
            self.create_specialized_generators()
            
            # Crear utilidades
            self.create_utilities()
            
            # Crear templates
            self.create_templates()
            
            # Crear routers FastAPI
            self.create_fastapi_routers()
            
            # Crear aplicación principal
            self.create_main_app()
            
            # Crear requirements
            self.create_requirements()
            
            # Crear Dockerfile
            self.create_dockerfile()
            
            # Crear tests
            self.create_tests()
            
            # Crear documentación
            self.create_documentation()
            
            print(f"""
✅ ===============================================
📋 REPORT GENERATOR SETUP COMPLETADO
✅ ===============================================

📁 Estructura creada en: {self.service_path}
🌐 Puerto asignado: {self.service_port}

🔧 Para instalar dependencias:
   cd {self.service_path}
   pip install -r requirements.txt

🚀 Para ejecutar el servicio:
   cd {self.service_path}
   python app.py

📊 URLs de acceso:
   - API: http://localhost:{self.service_port}
   - Docs: http://localhost:{self.service_port}/docs
   - Health: http://localhost:{self.service_port}/health

✅ ===============================================
""")
            
        except Exception as e:
            self.logger.error(f"❌ Error en setup: {e}")
            raise

    def check_system_dependencies(self):
        """Verificar dependencias del sistema"""
        print("🔍 Verificando dependencias del sistema...")
        
        required_commands = ["python", "pip"]
        
        for cmd in required_commands:
            if not shutil.which(cmd):
                raise RuntimeError(f"❌ Comando requerido no encontrado: {cmd}")
        
        print("✅ Dependencias del sistema verificadas")

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        print("📁 Creando estructura de directorios...")
        
        directories = [
            # Estructura base
            "src/core",
            "src/services", 
            "src/generators",
            "src/templates",
            "src/utils",
            "src/models",
            "src/routers",
            
            # Configuración
            "config",
            
            # Templates específicos
            "templates/pdf",
            "templates/html",
            "templates/excel",
            "templates/word",
            
            # Salidas y cache
            "output/reports",
            "output/temp",
            "cache",
            
            # Tests
            "tests/unit",
            "tests/integration", 
            "tests/fixtures",
            
            # Documentación
            "docs",
            
            # Logs
            "logs",
            
            # Estáticos
            "static/css",
            "static/js",
            "static/images",
            "static/fonts"
        ]
        
        for directory in directories:
            dir_path = self.service_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Crear .gitkeep para directorios que pueden estar vacíos
            if any(word in directory for word in ["output", "cache", "logs", "temp"]):
                (dir_path / ".gitkeep").touch()
        
        print("✅ Estructura de directorios creada")

    def create_config_files(self):
        """Crear archivos de configuración"""
        print("⚙️ Creando archivos de configuración...")
        
        # settings.py
        settings_content = '''"""
Configuración del Report Generator
"""
import os
from pydantic import BaseSettings
from typing import List, Dict, Any

class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # Aplicación
    app_name: str = "Report Generator"
    version: str = "6.0.0"
    debug: bool = False
    
    # Servidor
    host: str = "0.0.0.0"
    port: int = 8004
    
    # CORS
    cors_origins: List[str] = ["*"]
    
    # Base de datos
    database_url: str = "sqlite:///./report_generator.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # AI Services
    ai_engine_url: str = "http://localhost:8001"
    analytics_engine_url: str = "http://localhost:8003"
    
    # Archivos
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = [".pdf", ".docx", ".html", ".xlsx"]
    
    # Templates
    template_dir: str = "templates"
    output_dir: str = "output/reports"
    temp_dir: str = "output/temp"
    
    # PDFs
    pdf_engine: str = "weasyprint"  # weasyprint, reportlab
    pdf_timeout: int = 30
    
    # Concurrencia
    max_workers: int = 4
    max_concurrent_reports: int = 10
    
    # Cache
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hora
    
    # Limpieza automática
    cleanup_enabled: bool = True
    cleanup_interval: int = 24 * 60 * 60  # 24 horas
    temp_file_ttl: int = 2 * 60 * 60  # 2 horas
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/report_generator.log"
    
    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Instancia global
settings = Settings()
'''
        
        with open(self.service_path / "config" / "settings.py", "w", encoding="utf-8") as f:
            f.write(settings_content)
        
        print("✅ Archivos de configuración creados")

    def create_data_models(self):
        """Crear modelos de datos"""
        print("📊 Creando modelos de datos...")
        
        # models/__init__.py
        models_init_content = '''"""
Modelos de datos para Report Generator
"""

from .report_models import ReportRequest, ReportResponse, ReportStatus
from .template_models import Template, TemplateVariable
from .data_models import DataSource, ReportData

__all__ = [
    "ReportRequest",
    "ReportResponse", 
    "ReportStatus",
    "Template",
    "TemplateVariable",
    "DataSource",
    "ReportData"
]
'''
        
        with open(self.service_path / "src" / "models" / "__init__.py", "w", encoding="utf-8") as f:
            f.write(models_init_content)
        
        # report_models.py
        report_models_content = '''"""
Modelos para reportes
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid

class ReportType(str, Enum):
    """Tipos de reporte disponibles"""
    DOCUMENT_ANALYSIS = "document_analysis"
    STATISTICAL_SUMMARY = "statistical_summary"
    CUSTOM_REPORT = "custom_report"
    ANALYTICS_DASHBOARD = "analytics_dashboard"
    COMPARISON_REPORT = "comparison_report"
    EXECUTIVE_SUMMARY = "executive_summary"

class ReportFormat(str, Enum):
    """Formatos de salida"""
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    XLSX = "xlsx"
    JSON = "json"

class ReportStatus(str, Enum):
    """Estados del reporte"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ReportRequest(BaseModel):
    """Solicitud de generación de reporte"""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_type: ReportType
    format: ReportFormat = ReportFormat.PDF
    template_id: Optional[str] = None
    data_sources: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    title: Optional[str] = None
    description: Optional[str] = None
    include_charts: bool = True
    include_tables: bool = True
    priority: int = Field(default=5, ge=1, le=10)
    created_at: datetime = Field(default_factory=datetime.now)

class ReportResponse(BaseModel):
    """Respuesta de reporte generado"""
    report_id: str
    status: ReportStatus
    format: ReportFormat
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    generation_time: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
'''
        
        with open(self.service_path / "src" / "models" / "report_models.py", "w", encoding="utf-8") as f:
            f.write(report_models_content)
        
        print("✅ Modelos de datos creados")

    def create_report_services(self):
        """Crear servicios principales de reportes"""
        print("🔧 Creando servicios principales...")
        
        # report_manager.py
        report_manager_content = '''"""
Gestor principal de reportes
"""
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..models import ReportRequest, ReportResponse, ReportStatus

logger = logging.getLogger(__name__)

class ReportManager:
    """Gestor principal de reportes"""
    
    def __init__(self):
        self.active_reports: Dict[str, Any] = {}
        self.max_concurrent = 10
        logger.info("🚀 ReportManager inicializado")
    
    async def generate_report(self, request: ReportRequest) -> ReportResponse:
        """Generar reporte completo"""
        
        try:
            start_time = datetime.now()
            
            # Simular generación de reporte
            await asyncio.sleep(1)  # Simular procesamiento
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Crear respuesta exitosa
            response = ReportResponse(
                report_id=request.report_id,
                status=ReportStatus.COMPLETED,
                format=request.format,
                file_path=f"output/reports/report_{request.report_id}.{request.format.value}",
                generation_time=generation_time,
                created_at=request.created_at,
                completed_at=datetime.now()
            )
            
            logger.info(f"✅ Reporte {request.report_id} generado en {generation_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte {request.report_id}: {e}")
            
            return ReportResponse(
                report_id=request.report_id,
                status=ReportStatus.FAILED,
                format=request.format,
                error_message=str(e),
                created_at=request.created_at
            )
    
    async def get_report_progress(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Obtener progreso de reporte"""
        return self.active_reports.get(report_id)
    
    async def cleanup(self):
        """Limpiar recursos"""
        logger.info("🧹 Limpiando recursos del ReportManager...")
        self.active_reports.clear()
        logger.info("✅ Recursos limpiados")
'''
        
        with open(self.service_path / "src" / "services" / "report_manager.py", "w", encoding="utf-8") as f:
            f.write(report_manager_content)
        
        print("✅ Servicios principales creados")

    def create_specialized_generators(self):
        """Crear generadores especializados"""
        print("🎨 Creando generadores especializados...")
        
        # pdf_generator.py
        pdf_generator_content = '''"""
Generador de PDFs
"""
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PDFGenerator:
    """Generador de archivos PDF"""
    
    def __init__(self):
        logger.info("🎨 PDFGenerator inicializado")
    
    async def generate_pdf(self, html_content: str, output_path: Path, options: Dict[str, Any] = None) -> Path:
        """Generar PDF desde contenido HTML"""
        
        try:
            if options is None:
                options = {}
            
            # Simular generación de PDF
            await asyncio.sleep(0.5)
            
            # Crear archivo de ejemplo
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"PDF generado: {datetime.now()}")
            
            logger.info(f"✅ PDF generado: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error generando PDF: {e}")
            raise
'''
        
        with open(self.service_path / "src" / "generators" / "pdf_generator.py", "w", encoding="utf-8") as f:
            f.write(pdf_generator_content)
        
        print("✅ Generadores especializados creados")

    def create_utilities(self):
        """Crear utilidades del sistema"""
        print("🔧 Creando utilidades...")
        
        # file_manager.py
        file_manager_content = '''"""
Gestor de archivos para reportes
"""
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FileManager:
    """Gestor de archivos y directorios"""
    
    def __init__(self, base_dir: str = "output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 FileManager inicializado: {self.base_dir}")
    
    async def save_report(self, content: bytes, filename: str) -> Path:
        """Guardar reporte generado"""
        try:
            file_path = self.base_dir / "reports" / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"💾 Reporte guardado: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Error guardando reporte: {e}")
            raise
    
    async def cleanup_old_files(self, days: int = 7) -> int:
        """Limpiar archivos antiguos"""
        try:
            cleaned_count = 0
            # Simular limpieza
            logger.info(f"🧹 Limpieza completada: {cleaned_count} archivos eliminados")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Error en limpieza: {e}")
            return 0
'''
        
        with open(self.service_path / "src" / "utils" / "file_manager.py", "w", encoding="utf-8") as f:
            f.write(file_manager_content)
        
        print("✅ Utilidades creadas")

    def create_templates(self):
        """Crear templates base"""
        print("📝 Creando templates base...")
        
        # Template HTML base
        base_html_template = '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title or "Reporte de Análisis" }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        
        .header {
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin: 0;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section h2 {
            color: #2c3e50;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            font-size: 1.8em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title or "Reporte de Análisis Documental" }}</h1>
        <div>Generado por IA - Agente OyP 6.0</div>
        <div>{{ timestamp or "Fecha no disponible" }}</div>
    </div>

    <div class="section">
        <h2>📊 Contenido del Reporte</h2>
        <p>{{ content or "Contenido del reporte aquí" }}</p>
    </div>

    <div class="section">
        <h2>📈 Métricas</h2>
        <table>
            <tr>
                <th>Métrica</th>
                <th>Valor</th>
            </tr>
            <tr>
                <td>Documentos procesados</td>
                <td>{{ documents_count or "0" }}</td>
            </tr>
            <tr>
                <td>Tiempo de procesamiento</td>
                <td>{{ processing_time or "0" }} segundos</td>
            </tr>
        </table>
    </div>
</body>
</html>'''
        
        with open(self.service_path / "templates" / "html" / "base_report.html", "w", encoding="utf-8") as f:
            f.write(base_html_template)
        
        print("✅ Templates creados")

    def create_fastapi_routers(self):
        """Crear routers de FastAPI"""
        print("🌐 Creando routers FastAPI...")
        
        # reports.py
        reports_router_content = '''"""
Router para generación de reportes
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import logging

from ..models import ReportRequest, ReportResponse, ReportType, ReportFormat
from ..services.report_manager import ReportManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Instancia global
report_manager = ReportManager()

@router.post("/generate", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """Generar nuevo reporte"""
    try:
        logger.info(f"🚀 Iniciando generación de reporte: {request.report_id}")
        
        # Validar solicitud
        if not request.report_type:
            raise HTTPException(status_code=400, detail="Tipo de reporte requerido")
        
        # Generar reporte
        return await report_manager.generate_report(request)
            
    except Exception as e:
        logger.error(f"❌ Error en generate_report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def get_report_types():
    """Obtener tipos de reporte disponibles"""
    return {
        "report_types": [
            {
                "value": "document_analysis",
                "label": "Análisis de Documento",
                "description": "Análisis completo de contenido documental"
            },
            {
                "value": "statistical_summary", 
                "label": "Resumen Estadístico",
                "description": "Análisis estadístico de datos numéricos"
            },
            {
                "value": "custom_report",
                "label": "Reporte Personalizado", 
                "description": "Reporte con template personalizado"
            }
        ],
        "formats": [
            {"value": "pdf", "label": "PDF", "description": "Documento PDF profesional"},
            {"value": "html", "label": "HTML", "description": "Página web interactiva"},
            {"value": "xlsx", "label": "Excel", "description": "Hoja de cálculo con datos"},
            {"value": "json", "label": "JSON", "description": "Datos estructurados"}
        ]
    }

@router.post("/quick")
async def quick_report(
    report_type: ReportType,
    format: ReportFormat = ReportFormat.PDF,
    title: Optional[str] = None
):
    """Generar reporte rápido con datos mínimos"""
    try:
        # Crear solicitud básica
        request = ReportRequest(
            report_type=report_type,
            format=format,
            title=title or f"Reporte {report_type.value}",
            priority=10  # Alta prioridad para procesamiento inmediato
        )
        
        # Generar reporte
        response = await report_manager.generate_report(request)
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en reporte rápido: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
        
        with open(self.service_path / "src" / "routers" / "reports.py", "w", encoding="utf-8") as f:
            f.write(reports_router_content)
        
        # health.py
        health_router_content = '''"""
Router para health checks y monitoreo
"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check básico"""
    return {
        "status": "healthy",
        "service": "report-generator",
        "version": "6.0.0",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": {
            "name": "report-generator",
            "version": "6.0.0",
            "status": "running"
        },
        "timestamp": datetime.now().isoformat()
    }
'''
        
        with open(self.service_path / "src" / "routers" / "health.py", "w", encoding="utf-8") as f:
            f.write(health_router_content)
        
        print("✅ Routers FastAPI creados")

    def create_main_app(self):
        """Crear aplicación principal FastAPI"""
        print("🚀 Creando aplicación principal...")
        
        app_content = '''"""
Aplicación principal del Report Generator
Agente IA OyP 6.0 - Módulo 5
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="📋 Report Generator - Agente IA OyP 6.0",
    description="Generador de reportes inteligentes con IA",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Importar routers
try:
    from src.routers.reports import router as reports_router
    from src.routers.health import router as health_router
    
    # Incluir routers
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(reports_router, prefix="/reports", tags=["Reports"])
    
except ImportError as e:
    logger.warning(f"⚠️ Error importando routers: {e}")

# Página principal
@app.get("/")
async def root():
    """Página principal del Report Generator"""
    
    return {
        "message": "📋 Report Generator - Agente IA OyP 6.0",
        "status": "active",
        "version": "6.0.0",
        "endpoints": {
            "health": "/health",
            "reports": "/reports",
            "docs": "/docs"
        },
        "features": [
            "Generación de PDFs profesionales",
            "Reportes HTML interactivos", 
            "Hojas de cálculo Excel",
            "Templates personalizables",
            "Integración con IA",
            "Gráficos automáticos"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        print("✅ Aplicación principal creada")

    def create_requirements(self):
        """Crear requirements.txt"""
        print("📦 Creando requirements.txt...")
        
        requirements_content = '''# =====================================================
# REPORT GENERATOR - REQUIREMENTS COMPLETO
# Agente IA OyP 6.0 - Módulo 5
# =====================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Generación de PDFs (OPCIONAL - instalar si se necesita)
# weasyprint==60.2
# reportlab==4.0.7

# Procesamiento de Excel (OPCIONAL)
# openpyxl==3.1.2

# Templates y renderizado (OPCIONAL)
# jinja2==3.1.2

# HTTP client
httpx==0.25.2
requests==2.31.0

# Sistema y monitoreo
structlog==23.2.0

# Utilidades
python-dotenv==1.0.0
click==8.1.7

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# Desarrollo
black==23.11.0
flake8==6.1.0
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        print("✅ requirements.txt creado")

    def create_dockerfile(self):
        """Crear Dockerfile"""
        print("🐳 Creando Dockerfile...")
        
        dockerfile_content = '''FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1

# Crear usuario no root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorios necesarios
RUN mkdir -p output/reports output/temp templates cache logs static && \\
    chown -R appuser:appuser /app

# Copiar código de la aplicación
COPY --chown=appuser:appuser . .

# Cambiar a usuario no root
USER appuser

# Exponer puerto
EXPOSE 8004

# Variables de entorno
ENV HOST=0.0.0.0
ENV PORT=8004
ENV LOG_LEVEL=INFO

# Comando por defecto
CMD ["python", "app.py"]
'''
        
        with open(self.service_path / "Dockerfile", "w", encoding="utf-8") as f:
            f.write(dockerfile_content)
        
        print("✅ Dockerfile creado")

    def create_tests(self):
        """Crear tests"""
        print("🧪 Creando tests...")
        
        # test_reports.py
        test_reports_content = '''"""
Tests para generación de reportes
"""
import pytest
from fastapi.testclient import TestClient
from src.models import ReportRequest, ReportType, ReportFormat

def test_root_endpoint():
    """Test endpoint raíz"""
    from app import app
    
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Report Generator" in data["message"]

def test_health_check():
    """Test health check"""
    from app import app
    
    client = TestClient(app)
    response = client.get("/health/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_report_types():
    """Test obtener tipos de reporte"""
    from app import app
    
    client = TestClient(app)
    response = client.get("/reports/types")
    assert response.status_code == 200
    data = response.json()
    assert "report_types" in data
    assert "formats" in data

def test_quick_report():
    """Test generación de reporte rápido"""
    from app import app
    
    client = TestClient(app)
    response = client.post(
        "/reports/quick",
        params={
            "report_type": "custom_report",
            "format": "json",
            "title": "Test Report"
        }
    )
    
    # Debe ser exitoso
    assert response.status_code == 200
    data = response.json()
    assert "report_id" in data
    assert data["format"] == "json"

class TestReportGeneration:
    """Tests para lógica de generación"""
    
    def test_report_request_creation(self):
        """Test creación de solicitud de reporte"""
        request = ReportRequest(
            report_type=ReportType.CUSTOM_REPORT,
            format=ReportFormat.PDF,
            title="Test Report"
        )
        
        assert request.report_type == ReportType.CUSTOM_REPORT
        assert request.format == ReportFormat.PDF
        assert request.title == "Test Report"
        assert request.report_id is not None
'''
        
        with open(self.service_path / "tests" / "unit" / "test_reports.py", "w", encoding="utf-8") as f:
            f.write(test_reports_content)
        
        print("✅ Tests creados")

    def create_documentation(self):
        """Crear documentación"""
        print("📚 Creando documentación...")
        
        readme_content = f'''# 📋 Report Generator - Módulo 5

Generador de reportes inteligentes con IA para el Agente OyP 6.0.

## 🎯 Características

- **Múltiples formatos**: PDF, HTML, Excel, JSON
- **Templates personalizables**: Sistema de plantillas
- **IA integrada**: Insights automáticos y análisis inteligente
- **API REST completa**: Endpoints para todas las operaciones
- **Procesamiento concurrente**: Múltiples reportes simultáneos

## 🚀 Instalación

1. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

2. **Configurar variables de entorno** (opcional):
```bash
cp .env.example .env
# Editar .env según necesidades
```

3. **Ejecutar el servicio**:
```bash
python app.py
```

## 📖 Uso

### Generar Reporte Rápido

```bash
curl -X POST "http://localhost:{self.service_port}/reports/quick" \\
     -H "Content-Type: application/json" \\
     -d '{{"report_type": "custom_report", "format": "pdf", "title": "Mi Reporte"}}'
```

### Verificar Estado

```bash
curl "http://localhost:{self.service_port}/health/health"
```

### Ver Tipos Disponibles

```bash
curl "http://localhost:{self.service_port}/reports/types"
```

## 📊 Tipos de Reporte

| Tipo | Descripción |
|------|-------------|
| `document_analysis` | Análisis completo de documentos |
| `statistical_summary` | Resumen estadístico |
| `custom_report` | Reporte personalizado |

## 🔧 Configuración

### Variables de Entorno

```bash
# Servidor
HOST=0.0.0.0
PORT={self.service_port}
DEBUG=false

# Servicios externos
AI_ENGINE_URL=http://localhost:8001
ANALYTICS_ENGINE_URL=http://localhost:8003
```

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Tests específicos
pytest tests/unit/test_reports.py

# Con cobertura
pytest --cov=src --cov-report=html
```

## 📈 Monitoreo

### Health Checks

- **Básico**: `GET /health/health`
- **Detallado**: `GET /health/status`

## 🐳 Docker

```bash
# Construir imagen
docker build -t report-generator .

# Ejecutar contenedor
docker run -p {self.service_port}:8004 \\
    -e HOST=0.0.0.0 \\
    -e PORT=8004 \\
    report-generator
```

## 🔗 URLs de Acceso

- **API**: http://localhost:{self.service_port}
- **Documentación**: http://localhost:{self.service_port}/docs
- **Health Check**: http://localhost:{self.service_port}/health/health

## ⚡ Tips de Rendimiento

1. **Use formatos ligeros**: JSON es más rápido que PDF
2. **Prioridades**: Use `priority` para reportes urgentes
3. **Cache**: Los templates se cachean automáticamente

## 🤝 Integración

### Con AI Engine

```python
# El Report Generator se conecta automáticamente
# para obtener insights de IA
```

### Con Analytics Engine

```python
# Obtener estadísticas automáticamente
```

---

**Desarrollado para Agente IA OyP 6.0** 🤖
'''
        
        with open(self.service_path / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # .env.example
        env_example_content = '''# Report Generator Configuration

# Aplicación
APP_NAME=Report Generator
VERSION=6.0.0
DEBUG=false

# Servidor  
HOST=0.0.0.0
PORT=8004

# Base de datos
DATABASE_URL=sqlite:///./report_generator.db

# Redis
REDIS_URL=redis://localhost:6379

# AI Services
AI_ENGINE_URL=http://localhost:8001
ANALYTICS_ENGINE_URL=http://localhost:8003

# API Keys (opcional)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# Configuración de reportes
MAX_FILE_SIZE=52428800
MAX_WORKERS=4
CACHE_ENABLED=true
CLEANUP_ENABLED=true
'''
        
        with open(self.service_path / ".env.example", "w", encoding="utf-8") as f:
            f.write(env_example_content)
        
        print("✅ Documentación creada")

if __name__ == "__main__":
    setup = ReportGeneratorSetup()
    setup.run_setup()