"""
Generador de Reportes - Servicio Principal
Puerto: 8004
Archivo: services/report-generator/src/main.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
import os
import json
import logging
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import io

# Generación de PDFs
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("⚠️ ReportLab no disponible - generación PDF limitada")

# Templates
try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("⚠️ Jinja2 no disponible - templates limitados")

import pandas as pd
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================================
# CONFIGURACIÓN DE DIRECTORIOS
# ===================================================================

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "data" / "reports"
TEMPLATES_DIR = BASE_DIR / "templates"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# MODELOS DE DATOS
# ===================================================================

class ReportRequest(BaseModel):
    title: str
    subtitle: Optional[str] = None
    author: str = "Agente IA OyP 6.0"
    data: Optional[Dict[str, Any]] = None
    sections: List[Dict[str, Any]] = []
    template: str = "standard"
    format: str = "pdf"  # pdf, html, json
    include_charts: bool = True
    include_tables: bool = True

class Section(BaseModel):
    title: str
    content: str
    type: str = "text"  # text, table, chart, analysis
    data: Optional[Dict[str, Any]] = None

class QuickReportRequest(BaseModel):
    title: str
    content: str
    format: str = "pdf"
    author: str = "Agente IA OyP 6.0"

# ===================================================================
# APLICACIÓN FASTAPI
# ===================================================================

app = FastAPI(
    title="📋 Generador de Reportes - Agente IA OyP 6.0",
    description="Generador automático de reportes profesionales en PDF, HTML y JSON",
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

# ===================================================================
# CLASE PRINCIPAL DEL SERVICIO
# ===================================================================

class ReportGeneratorService:
    """Generador de reportes con capacidades completas"""
    
    def __init__(self):
        self.generated_reports = {}
        self.templates = self.setup_templates()
        logger.info("✅ Report Generator Service inicializado")
    
    def setup_templates(self) -> Dict[str, Any]:
        """Configurar templates predefinidos"""
        return {
            "standard": {
                "name": "Reporte Estándar",
                "description": "Template básico para reportes generales",
                "sections": ["header", "summary", "content", "conclusions"]
            },
            "analytics": {
                "name": "Reporte de Analytics",
                "description": "Template para análisis de datos",
                "sections": ["header", "executive_summary", "data_analysis", "visualizations", "insights", "recommendations"]
            },
            "document_analysis": {
                "name": "Análisis Documental",
                "description": "Template para análisis de documentos",
                "sections": ["header", "document_info", "content_analysis", "key_findings", "summary"]
            }
        }
    
    async def generate_report(self, request: ReportRequest) -> Dict[str, Any]:
        """Generar reporte completo"""
        try:
            report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Construir contenido del reporte
            report_content = await self.build_report_content(request)
            
            # Generar según formato
            if request.format == "pdf":
                report_data = await self.generate_pdf_report(report_content, request)
            elif request.format == "html":
                report_data = await self.generate_html_report(report_content, request)
            elif request.format == "json":
                report_data = await self.generate_json_report(report_content, request)
            else:
                raise ValueError(f"Formato {request.format} no soportado")
            
            # Guardar reporte
            self.generated_reports[report_id] = {
                "id": report_id,
                "title": request.title,
                "format": request.format,
                "generated_at": datetime.now().isoformat(),
                "size": len(str(report_data.get("content", ""))),
                "request": request.dict()
            }
            
            return {
                "report_id": report_id,
                "title": request.title,
                "format": request.format,
                "generated_at": datetime.now().isoformat(),
                "status": "completed",
                **report_data
            }
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte: {e}")
            raise HTTPException(status_code=500, detail=f"Error generando reporte: {str(e)}")
    
    async def build_report_content(self, request: ReportRequest) -> Dict[str, Any]:
        """Construir contenido estructurado del reporte"""
        try:
            content = {
                "metadata": {
                    "title": request.title,
                    "subtitle": request.subtitle,
                    "author": request.author,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "template": request.template
                },
                "sections": [],
                "statistics": {}
            }
            
            # Procesar secciones del request
            for section_data in request.sections:
                processed_section = await self.process_section(section_data)
                content["sections"].append(processed_section)
            
            # Generar secciones automáticas si hay datos
            if request.data:
                auto_sections = await self.generate_auto_sections(request.data, request.template)
                content["sections"].extend(auto_sections)
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Error construyendo contenido: {e}")
            raise ValueError(f"Error construyendo contenido: {str(e)}")
    
    async def process_section(self, section_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar una sección individual"""
        try:
            processed = {
                "title": section_data.get("title", "Sin título"),
                "type": section_data.get("type", "text"),
                "content": section_data.get("content", ""),
                "data": section_data.get("data", {})
            }
            
            # Procesamiento específico por tipo
            if processed["type"] == "table" and processed["data"]:
                processed["formatted_table"] = self.format_table_data(processed["data"])
            elif processed["type"] == "analysis" and processed["data"]:
                processed["analysis_summary"] = self.generate_analysis_summary(processed["data"])
            
            return processed
            
        except Exception as e:
            logger.error(f"❌ Error procesando sección: {e}")
            return {"title": "Error", "content": f"Error procesando sección: {str(e)}"}
    
    async def generate_auto_sections(self, data: Dict[str, Any], template: str) -> List[Dict[str, Any]]:
        """Generar secciones automáticas basadas en datos"""
        try:
            sections = []
            
            # Sección de resumen de datos
            if "statistics" in data or "summary" in data:
                sections.append({
                    "title": "Resumen Ejecutivo",
                    "type": "analysis",
                    "content": self.generate_executive_summary(data),
                    "data": data.get("statistics", {})
                })
            
            # Sección de análisis de datos
            if "analysis" in data:
                sections.append({
                    "title": "Análisis de Datos",
                    "type": "analysis",
                    "content": self.format_analysis_results(data["analysis"]),
                    "data": data["analysis"]
                })
            
            # Sección de tablas
            if "tables" in data or "dataframe" in data:
                sections.append({
                    "title": "Datos Tabulares",
                    "type": "table",
                    "content": "Presentación de los datos en formato tabular para análisis detallado.",
                    "data": data.get("tables", data.get("dataframe", []))
                })
            
            # Sección de hallazgos clave
            if "insights" in data or "findings" in data:
                sections.append({
                    "title": "Hallazgos Principales",
                    "type": "text",
                    "content": data.get("insights", data.get("findings", "")),
                    "data": {}
                })
            
            # Sección de recomendaciones
            if "recommendations" in data:
                sections.append({
                    "title": "Recomendaciones",
                    "type": "text",
                    "content": data["recommendations"],
                    "data": {}
                })
            
            return sections
            
        except Exception as e:
            logger.error(f"❌ Error generando secciones automáticas: {e}")
            return []
    
    def generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generar resumen ejecutivo"""
        try:
            summary_parts = []
            
            # Información del dataset
            if "statistics" in data:
                stats = data["statistics"]
                if "rows" in stats and "columns" in stats:
                    summary_parts.append(f"**Datos analizados:** {stats['rows']:,} registros con {stats['columns']} variables.")
                
                if "data_quality" in stats:
                    quality = stats["data_quality"]
                    completeness = quality.get("completeness", 0)
                    summary_parts.append(f"**Calidad de datos:** {completeness:.1f}% de completitud.")
            
            # Correlaciones significativas
            if "correlations" in data:
                corr_data = data["correlations"]
                if "strong_correlations" in corr_data:
                    strong_count = len(corr_data["strong_correlations"])
                    if strong_count > 0:
                        summary_parts.append(f"**Correlaciones identificadas:** {strong_count} correlaciones estadísticamente significativas.")
            
            # Clustering
            if "clustering" in data:
                cluster_data = data["clustering"]
                if "optimal_clusters" in cluster_data:
                    clusters = cluster_data["optimal_clusters"]
                    summary_parts.append(f"**Segmentación:** Los datos se organizan naturalmente en {clusters} grupos distintos.")
            
            # Outliers
            if "outliers" in data:
                outlier_data = data["outliers"]
                total_outliers = sum(info.get("count", 0) for info in outlier_data.values() if isinstance(info, dict))
                if total_outliers > 0:
                    summary_parts.append(f"**Valores atípicos:** Se detectaron {total_outliers} valores que requieren atención especial.")
            
            if not summary_parts:
                summary_parts.append("Se ha completado el análisis de los datos proporcionados con resultados satisfactorios.")
            
            return "\n\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"❌ Error generando resumen ejecutivo: {e}")
            return "Resumen ejecutivo no disponible debido a un error en el procesamiento."
    
    def format_analysis_results(self, analysis: Dict[str, Any]) -> str:
        """Formatear resultados de análisis"""
        try:
            content_parts = []
            
            # Estadísticas descriptivas
            if "descriptive_stats" in analysis:
                content_parts.append("## Estadísticas Descriptivas\n")
                content_parts.append("Se realizó un análisis estadístico completo de todas las variables numéricas y categóricas del dataset.\n")
            
            # Análisis de correlaciones
            if "correlations" in analysis:
                content_parts.append("## Análisis de Correlaciones\n")
                corr = analysis["correlations"]
                if "strong_correlations" in corr and corr["strong_correlations"]:
                    content_parts.append("**Correlaciones significativas identificadas:**\n")
                    for i, corr_item in enumerate(corr["strong_correlations"][:5], 1):
                        var1 = corr_item["variable_1"]
                        var2 = corr_item["variable_2"]
                        corr_val = corr_item["correlation"]
                        direction = "positiva" if corr_val > 0 else "negativa"
                        content_parts.append(f"{i}. **{var1}** y **{var2}**: Correlación {direction} fuerte ({corr_val:.3f})")
                else:
                    content_parts.append("No se encontraron correlaciones estadísticamente significativas entre las variables.\n")
            
            # Análisis de clustering
            if "clustering" in analysis:
                content_parts.append("\n## Análisis de Segmentación (Clustering)\n")
                clustering = analysis["clustering"]
                clusters = clustering.get("optimal_clusters", 0)
                content_parts.append(f"El análisis identificó **{clusters} grupos principales** en los datos con las siguientes características:\n")
                
                if "cluster_analysis" in clustering:
                    for cluster_id, cluster_info in clustering["cluster_analysis"].items():
                        size = cluster_info.get("size", 0)
                        percentage = cluster_info.get("percentage", 0)
                        content_parts.append(f"- **{cluster_id.replace('_', ' ').title()}**: {size} elementos ({percentage:.1f}% del total)")
            
            # Detección de outliers
            if "outliers" in analysis:
                content_parts.append("\n## Detección de Valores Atípicos\n")
                outliers = analysis["outliers"]
                outlier_vars = [var for var, info in outliers.items() if info.get("count", 0) > 0]
                
                if outlier_vars:
                    content_parts.append("Se detectaron valores atípicos en las siguientes variables:\n")
                    for var in outlier_vars[:5]:  # Máximo 5 variables
                        count = outliers[var]["count"]
                        percentage = outliers[var]["percentage"]
                        content_parts.append(f"- **{var}**: {count} valores atípicos ({percentage:.1f}%)")
                else:
                    content_parts.append("No se detectaron valores atípicos significativos en el dataset.\n")
            
            # Análisis de texto (si existe)
            if "text_analysis" in analysis:
                content_parts.append("\n## Análisis de Contenido Textual\n")
                text_analysis = analysis["text_analysis"]
                if "sentiment" in text_analysis:
                    sentiment = text_analysis["sentiment"]
                    content_parts.append(f"**Análisis de sentimientos**: {sentiment.get('sentiment', 'N/A')} (confianza: {sentiment.get('confidence', 0):.2f})")
                
                if "keywords" in text_analysis:
                    keywords = text_analysis["keywords"][:5]
                    if keywords:
                        content_parts.append(f"**Palabras clave principales**: {', '.join(keywords)}")
            
            return "\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"❌ Error formateando análisis: {e}")
            return "Los resultados del análisis no pudieron ser formateados correctamente."

# ===================================================================
# INSTANCIA GLOBAL DEL SERVICIO
# ===================================================================

# Crear instancia global del servicio
try:
    report_service = ReportGeneratorService()
    logger.info("✅ Servicio de generación de reportes inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error al inicializar el servicio de reportes: {e}")
    raise

# ===================================================================
# ENDPOINTS DE LA API
# ===================================================================

@app.get("/")
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "service": "Generador de Reportes",
        "version": "6.0.0",
        "status": "activo",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": [
            {"path": "/", "methods": ["GET"], "description": "Información del servicio"},
            {"path": "/health", "methods": ["GET"], "description": "Health check del servicio"},
            {"path": "/info", "methods": ["GET"], "description": "Información detallada del servicio"},
            {"path": "/generate", "methods": ["POST"], "description": "Generar un nuevo reporte"},
            {"path": "/quick-report", "methods": ["POST"], "description": "Generar un reporte rápido"},
            {"path": "/templates", "methods": ["GET"], "description": "Listar templates disponibles"},
            {"path": "/reports", "methods": ["GET"], "description": "Listar reportes generados"},
            {"path": "/reports/{report_id}", "methods": ["GET"], "description": "Obtener información de un reporte específico"},
            {"path": "/reports/{report_id}", "methods": ["DELETE"], "description": "Eliminar un reporte del caché"},
            {"path": "/stats", "methods": ["GET"], "description": "Estadísticas del servicio"},
            {"path": "/test", "methods": ["GET"], "description": "Probar la generación de reportes"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check del servicio"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "Generador de Reportes",
        "version": "6.0.0",
        "dependencies": {
            "reportlab": "disponible" if REPORTLAB_AVAILABLE else "no disponible",
            "jinja2": "disponible" if JINJA2_AVAILABLE else "no disponible"
        }
    }

@app.get("/info")
async def service_info():
    """Información detallada del servicio"""
    return {
        "service": "Generador de Reportes",
        "version": "6.0.0",
        "description": "Servicio para la generación automática de reportes en múltiples formatos",
        "formats_supported": ["pdf", "html", "json"],
        "templates_available": list(report_service.templates.keys()),
        "reports_generated": len(report_service.generated_reports),
        "status": "activo",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate")
async def generate_report_endpoint(request: ReportRequest):
    """Generar reporte completo"""
    return await report_service.generate_report(request)

@app.post("/quick-report")
async def quick_report(request: QuickReportRequest):
    """Generar reporte rápido con contenido simple"""
    report_request = ReportRequest(
        title=request.title,
        content=request.content,
        author=request.author,
        format=request.format,
        sections=[{"title": "Contenido", "content": request.content, "type": "text"}]
    )
    return await report_service.generate_report(report_request)

@app.get("/templates")
async def list_templates():
    """Listar templates disponibles"""
    return {
        "templates": report_service.templates,
        "count": len(report_service.templates),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/reports")
async def list_reports():
    """Listar reportes generados"""
    return {
        "reports": [
            {
                "id": report_id,
                "title": report.get("title", "Sin título"),
                "format": report.get("format", "desconocido"),
                "generated_at": report.get("generated_at"),
                "size": report.get("size", 0)
            }
            for report_id, report in report_service.generated_reports.items()
        ],
        "count": len(report_service.generated_reports),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/reports/{report_id}")
async def get_report_info(report_id: str):
    """Obtener información de reporte específico"""
    if report_id not in report_service.generated_reports:
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    
    return report_service.generated_reports[report_id]

@app.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    """Eliminar reporte del caché"""
    if report_id not in report_service.generated_reports:
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    
    del report_service.generated_reports[report_id]
    return {"status": "deleted", "report_id": report_id}

@app.get("/stats")
async def get_generation_statistics():
    """Estadísticas del servicio de generación"""
    # Calcular estadísticas básicas
    formats = {}
    sizes = []
    
    for report in report_service.generated_reports.values():
        fmt = report.get("format", "desconocido")
        formats[fmt] = formats.get(fmt, 0) + 1
        sizes.append(report.get("size", 0))
    
    # Calcular estadísticas de tamaño
    size_stats = {
        "total": sum(sizes),
        "count": len(sizes),
        "avg": sum(sizes) / len(sizes) if sizes else 0,
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0
    }
    
    return {
        "reports_generated": len(report_service.generated_reports),
        "formats": formats,
        "size_stats": size_stats,
        "templates_available": len(report_service.templates),
        "service_uptime": str(datetime.now() - report_service.start_time),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/test")
async def test_report_generation():
    """Endpoint para probar la generación de reportes"""
    try:
        # Datos de prueba
        test_data = {
            "statistics": {
                "rows": 150,
                "columns": 8,
                "data_quality": {
                    "completeness": 97.5,
                    "duplicates": 3,
                    "missing_values": 12
                }
            },
            "analysis": {
                "descriptive_stats": {
                    "age": {"mean": 35.2, "min": 18, "max": 65, "std": 8.7},
                    "income": {"mean": 45000, "min": 25000, "max": 120000, "std": 15000}
                },
                "correlations": {
                    "strong_correlations": [
                        {"variable_1": "edad", "variable_2": "ingresos", "correlation": 0.78},
                        {"variable_1": "experiencia", "variable_2": "salario", "correlation": 0.82}
                    ]
                }
            }
        }
        
        # Crear solicitud de prueba
        request = ReportRequest(
            title="Reporte de Prueba",
            subtitle="Generado automáticamente para pruebas",
            author="Sistema Automatizado",
            data=test_data,
            format="pdf"
        )
        
        # Generar reporte
        result = await report_service.generate_report(request)
        
        return {
            "status": "success",
            "report_id": result.get("report_id"),
            "message": "Reporte de prueba generado exitosamente",
            "details": {
                "format": "pdf",
                "sections_generated": len(test_data.get("sections", [])),
                "data_points": test_data.get("statistics", {}).get("rows", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error en prueba de generación: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en prueba de generación: {str(e)}")

# ===================================================================
# INICIALIZACIÓN DEL SERVICIO
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)