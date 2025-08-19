#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📄 AGENTE IA OYP 6.0 - DOCUMENT-PROCESSOR SETUP
===============================================
Script específico para configurar el Document-Processor (Módulo 3)
Procesamiento inteligente de PDF, DOCX, TXT, OCR
Ejecutar desde: services/document-processor/
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class DocumentProcessorSetup:
    def __init__(self):
        self.service_path = Path.cwd()
        self.project_root = self.service_path.parent.parent
        self.venv_path = self.project_root / "venv"
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validar que estamos en el directorio correcto"""
        if not self.service_path.name == "document-processor":
            self.logger.error("❌ Debes ejecutar este script desde services/document-processor/")
            return False
        
        if not self.venv_path.exists():
            self.logger.error("❌ Entorno virtual no encontrado. Ejecuta primero setup_project.py")
            return False
        
        return True

    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.logger.info("📁 Creando estructura del Document-Processor...")
        
        directories = [
            "processors",
            "extractors", 
            "parsers",
            "routers",
            "services",
            "utils",
            "tests",
            "logs",
            "data",
            "data/uploads",
            "data/processed",
            "data/cache",
            "configs",
            "templates"
        ]
        
        for directory in directories:
            (self.service_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            python_dirs = ["processors", "extractors", "parsers", "routers", "services", "utils", "tests"]
            if any(directory.startswith(py_dir) for py_dir in python_dirs):
                init_file = self.service_path / directory / "__init__.py"
                init_file.touch()
        
        self.logger.info("✅ Estructura de directorios creada")

    def create_main_app(self):
        """Crear app.py principal"""
        self.logger.info("🔧 Creando app.py principal...")
        
        app_content = '''from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Routers
from routers import upload, extraction, parsing, conversion, analysis
from services.document_manager import DocumentManager
from services.text_extractor import TextExtractor
from services.ocr_service import OCRService
from processors.pdf_processor import PDFProcessor
from processors.docx_processor import DOCXProcessor
from processors.image_processor import ImageProcessor
from utils.config import Settings
from utils.logging_config import setup_logging

# Configuración global
settings = Settings()
document_manager = DocumentManager()
text_extractor = TextExtractor()
ocr_service = OCRService()

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    
    # Startup
    logger.info("📄 Iniciando Document-Processor...")
    
    try:
        # Crear directorios necesarios
        os.makedirs("data/uploads", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        
        # Inicializar servicios
        await document_manager.initialize()
        await text_extractor.initialize()
        await ocr_service.initialize()
        
        logger.info("✅ Document-Processor iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error iniciando Document-Processor: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Document-Processor...")
    await document_manager.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="📄 Document-Processor - Agente IA OyP 6.0",
    description="Procesamiento inteligente de documentos: PDF, DOCX, TXT, OCR",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="data"), name="static")

# Health check
@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy",
        "service": "document-processor",
        "version": "6.0.0",
        "supported_formats": ["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        "ocr_enabled": await ocr_service.is_available(),
        "processors_ready": document_manager.is_ready
    }

# Status detallado
@app.get("/status")
async def detailed_status():
    """Estado detallado del servicio"""
    return {
        "service": "document-processor",
        "version": "6.0.0",
        "supported_formats": {
            "documents": ["pdf", "docx", "txt", "rtf"],
            "images": ["png", "jpg", "jpeg", "tiff", "bmp"],
            "spreadsheets": ["xlsx", "xls", "csv"]
        },
        "capabilities": {
            "text_extraction": True,
            "ocr": await ocr_service.is_available(),
            "metadata_extraction": True,
            "format_conversion": True,
            "batch_processing": True
        },
        "statistics": await document_manager.get_statistics(),
        "storage": {
            "uploads_dir": "data/uploads",
            "processed_dir": "data/processed",
            "cache_dir": "data/cache"
        }
    }

# Información de formatos soportados
@app.get("/formats")
async def supported_formats():
    """Obtener formatos soportados"""
    return {
        "input_formats": {
            "documents": {
                "pdf": {"description": "Portable Document Format", "extractors": ["pdfplumber", "PyPDF2"]},
                "docx": {"description": "Microsoft Word Document", "extractors": ["python-docx"]},
                "txt": {"description": "Plain Text", "extractors": ["built-in"]},
                "rtf": {"description": "Rich Text Format", "extractors": ["striprtf"]}
            },
            "images": {
                "png": {"description": "Portable Network Graphics", "ocr": True},
                "jpg": {"description": "JPEG Image", "ocr": True},
                "jpeg": {"description": "JPEG Image", "ocr": True},
                "tiff": {"description": "Tagged Image File", "ocr": True},
                "bmp": {"description": "Bitmap Image", "ocr": True}
            },
            "spreadsheets": {
                "xlsx": {"description": "Microsoft Excel Spreadsheet", "extractors": ["openpyxl"]},
                "xls": {"description": "Microsoft Excel Legacy", "extractors": ["xlrd"]},
                "csv": {"description": "Comma Separated Values", "extractors": ["pandas"]}
            }
        },
        "output_formats": ["txt", "json", "markdown", "html", "csv"]
    }

# Incluir routers
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(extraction.router, prefix="/extract", tags=["Extraction"])
app.include_router(parsing.router, prefix="/parse", tags=["Parsing"])
app.include_router(conversion.router, prefix="/convert", tags=["Conversion"])
app.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8002,
        reload=settings.debug,
        log_level="info"
    )
'''
        
        with open(self.service_path / "app.py", "w", encoding="utf-8") as f:
            f.write(app_content)
        
        self.logger.info("✅ app.py creado")

    def create_requirements(self):
        """Crear requirements.txt específico"""
        self.logger.info("📦 Creando requirements.txt...")
        
        requirements_content = '''# ===============================================================================
# DOCUMENT-PROCESSOR - DEPENDENCIAS ESPECÍFICAS
# ===============================================================================

# FastAPI y servidor
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Procesamiento de PDF
PyPDF2==3.0.1
pdfplumber==0.9.0
pymupdf==1.23.8
pdfminer.six==20231228

# Procesamiento de DOCX
python-docx==1.1.0
python-pptx==0.6.23

# Procesamiento de Excel
openpyxl==3.1.2
xlrd==2.0.1
xlsxwriter==3.1.9

# OCR y procesamiento de imágenes
pytesseract==0.3.10
Pillow==10.1.0
opencv-python==4.8.1.78
pdf2image==1.16.3

# Procesamiento de texto
textract-py3
striprtf==0.0.26
python-magic==0.4.27

# Análisis y extracción
pandas==2.1.3
numpy==1.24.3
beautifulsoup4==4.12.2
lxml==4.9.3

# Utilidades y formatos
chardet==5.2.0
langdetect==1.0.9
unidecode==1.3.7

# Base de datos y caché
redis==5.0.1
s sqlalchemy==2.0.23

# Utilidades del sistema
python-dotenv==1.0.0
httpx==0.25.2
aiofiles==23.2.1
structlog==23.2.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Seguridad
python-magic==0.4.27
'''
        
        with open(self.service_path / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        self.logger.info("✅ requirements.txt creado")

    def create_text_extractor(self):
        """Crear servicio de extracción de texto"""
        self.logger.info("📝 Creando extractor de texto...")
        
        text_extractor_content = '''"""
Servicio de extracción de texto de múltiples formatos
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import aiofiles
import chardet
from dataclasses import dataclass

# Procesadores específicos
from processors.pdf_processor import PDFProcessor
from processors.docx_processor import DOCXProcessor
from processors.image_processor import ImageProcessor
from services.ocr_service import OCRService
from utils.file_utils import detect_file_type, get_file_info

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Resultado de extracción de texto"""
    text: str
    metadata: Dict[str, Any]
    pages: Optional[List[Dict]] = None
    format: str = ""
    extraction_method: str = ""
    confidence: float = 1.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class TextExtractor:
    """Extractor principal de texto para múltiples formatos"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.docx_processor = DOCXProcessor()
        self.image_processor = ImageProcessor()
        self.ocr_service = OCRService()
        self.is_ready = False
        
        # Mapeo de extensiones a procesadores
        self.format_handlers = {
            '.pdf': self._extract_from_pdf,
            '.docx': self._extract_from_docx,
            '.doc': self._extract_from_doc,
            '.txt': self._extract_from_txt,
            '.rtf': self._extract_from_rtf,
            '.png': self._extract_from_image,
            '.jpg': self._extract_from_image,
            '.jpeg': self._extract_from_image,
            '.tiff': self._extract_from_image,
            '.bmp': self._extract_from_image,
            '.xlsx': self._extract_from_excel,
            '.xls': self._extract_from_excel,
            '.csv': self._extract_from_csv
        }
    
    async def initialize(self):
        """Inicializar el extractor"""
        logger.info("🔧 Inicializando Text Extractor...")
        
        try:
            await self.pdf_processor.initialize()
            await self.docx_processor.initialize()
            await self.image_processor.initialize()
            await self.ocr_service.initialize()
            
            self.is_ready = True
            logger.info("✅ Text Extractor inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Text Extractor: {e}")
            raise
    
    async def extract_text(
        self, 
        file_path: Union[str, Path], 
        format_hint: Optional[str] = None,
        extraction_options: Optional[Dict] = None
    ) -> ExtractionResult:
        """Extraer texto de un archivo"""
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        # Detectar tipo de archivo
        file_info = await get_file_info(file_path)
        file_format = format_hint or file_info.get('extension', '').lower()
        
        logger.info(f"📄 Extrayendo texto de: {file_path.name} (formato: {file_format})")
        
        # Opciones por defecto
        options = extraction_options or {}
        
        try:
            # Seleccionar handler apropiado
            handler = self.format_handlers.get(file_format)
            if not handler:
                raise ValueError(f"Formato no soportado: {file_format}")
            
            # Extraer texto usando el handler específico
            result = await handler(file_path, options)
            result.format = file_format
            
            # Validar resultado
            if not result.text.strip():
                logger.warning(f"⚠️ No se extrajo texto de {file_path.name}")
            
            logger.info(f"✅ Texto extraído: {len(result.text)} caracteres")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo texto de {file_path.name}: {e}")
            return ExtractionResult(
                text="",
                metadata=file_info,
                format=file_format,
                extraction_method="error",
                confidence=0.0,
                errors=[str(e)]
            )
    
    async def _extract_from_pdf(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto de PDF"""
        return await self.pdf_processor.extract_text(file_path, options)
    
    async def _extract_from_docx(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto de DOCX"""
        return await self.docx_processor.extract_text(file_path, options)
    
    async def _extract_from_doc(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto de DOC (legacy)"""
        # Para archivos .doc, intentar conversión o usar textract
        try:
            import textract
            text = textract.process(str(file_path)).decode('utf-8')
            
            return ExtractionResult(
                text=text,
                metadata={"pages": 1, "format": "doc"},
                extraction_method="textract"
            )
        except Exception as e:
            logger.warning(f"⚠️ Error con textract para .doc: {e}")
            return ExtractionResult(
                text="",
                metadata={},
                extraction_method="doc_error",
                errors=[str(e)]
            )
    
    async def _extract_from_txt(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto de archivo de texto plano"""
        try:
            # Detectar codificación
            async with aiofiles.open(file_path, 'rb') as f:
                raw_data = await f.read()
            
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8')
            confidence = detected.get('confidence', 1.0)
            
            # Leer archivo con codificación detectada
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                text = await f.read()
            
            return ExtractionResult(
                text=text,
                metadata={
                    "encoding": encoding,
                    "lines": len(text.splitlines()),
                    "size": len(text)
                },
                extraction_method="direct",
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Error leyendo archivo de texto: {e}")
            return ExtractionResult(
                text="",
                metadata={},
                extraction_method="txt_error",
                errors=[str(e)]
            )
    
    async def _extract_from_rtf(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto de RTF"""
        try:
            from striprtf.striprtf import rtf_to_text
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                rtf_content = await f.read()
            
            text = rtf_to_text(rtf_content)
            
            return ExtractionResult(
                text=text,
                metadata={"format": "rtf"},
                extraction_method="striprtf"
            )
            
        except Exception as e:
            logger.error(f"❌ Error procesando RTF: {e}")
            return ExtractionResult(
                text="",
                metadata={},
                extraction_method="rtf_error",
                errors=[str(e)]
            )
    
    async def _extract_from_image(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto de imagen usando OCR"""
        return await self.ocr_service.extract_text(file_path, options)
    
    async def _extract_from_excel(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto de Excel"""
        try:
            import pandas as pd
            
            # Leer todas las hojas
            excel_file = pd.ExcelFile(file_path)
            all_text = []
            sheets_data = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_text = df.to_string(index=False)
                all_text.append(f"HOJA: {sheet_name}\n{sheet_text}")
                sheets_data[sheet_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "columns_names": list(df.columns)
                }
            
            combined_text = "\n\n".join(all_text)
            
            return ExtractionResult(
                text=combined_text,
                metadata={
                    "sheets": sheets_data,
                    "total_sheets": len(excel_file.sheet_names)
                },
                extraction_method="pandas"
            )
            
        except Exception as e:
            logger.error(f"❌ Error procesando Excel: {e}")
            return ExtractionResult(
                text="",
                metadata={},
                extraction_method="excel_error",
                errors=[str(e)]
            )
    
    async def _extract_from_csv(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto de CSV"""
        try:
            import pandas as pd
            
            # Detectar codificación
            async with aiofiles.open(file_path, 'rb') as f:
                raw_data = await f.read()
            
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8')
            
            # Leer CSV
            df = pd.read_csv(file_path, encoding=encoding)
            text = df.to_string(index=False)
            
            return ExtractionResult(
                text=text,
                metadata={
                    "rows": len(df),
                    "columns": len(df.columns),
                    "columns_names": list(df.columns),
                    "encoding": encoding
                },
                extraction_method="pandas"
            )
            
        except Exception as e:
            logger.error(f"❌ Error procesando CSV: {e}")
            return ExtractionResult(
                text="",
                metadata={},
                extraction_method="csv_error",
                errors=[str(e)]
            )
    
    async def batch_extract(
        self, 
        file_paths: List[Path], 
        extraction_options: Optional[Dict] = None
    ) -> List[ExtractionResult]:
        """Extraer texto de múltiples archivos en lote"""
        logger.info(f"📦 Procesamiento en lote: {len(file_paths)} archivos")
        
        tasks = [
            self.extract_text(file_path, extraction_options=extraction_options)
            for file_path in file_paths
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados y excepciones
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ExtractionResult(
                    text="",
                    metadata={"file": str(file_paths[i])},
                    extraction_method="batch_error",
                    errors=[str(result)]
                ))
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.text.strip())
        logger.info(f"✅ Lote completado: {successful}/{len(file_paths)} exitosos")
        
        return processed_results
    
    def get_supported_formats(self) -> List[str]:
        """Obtener lista de formatos soportados"""
        return list(self.format_handlers.keys())
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "text_extractor.py", "w", encoding="utf-8") as f:
            f.write(text_extractor_content)
        
        self.logger.info("✅ Extractor de texto creado")

    def create_pdf_processor(self):
        """Crear procesador de PDF"""
        self.logger.info("📄 Creando procesador de PDF...")
        
        pdf_processor_content = '''"""
Procesador especializado para archivos PDF
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader
from services.text_extractor import ExtractionResult

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Procesador especializado para archivos PDF"""
    
    def __init__(self):
        self.is_ready = False
        self.preferred_method = "pdfplumber"  # pdfplumber, pymupdf, pypdf2
    
    async def initialize(self):
        """Inicializar el procesador de PDF"""
        logger.info("🔧 Inicializando PDF Processor...")
        self.is_ready = True
        logger.info("✅ PDF Processor inicializado")
    
    async def extract_text(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto de PDF usando múltiples métodos"""
        
        method = options.get('method', self.preferred_method)
        
        try:
            if method == "pdfplumber":
                return await self._extract_with_pdfplumber(file_path, options)
            elif method == "pymupdf":
                return await self._extract_with_pymupdf(file_path, options)
            elif method == "pypdf2":
                return await self._extract_with_pypdf2(file_path, options)
            else:
                # Intentar con todos los métodos automáticamente
                return await self._extract_with_fallback(file_path, options)
                
        except Exception as e:
            logger.error(f"❌ Error procesando PDF {file_path.name}: {e}")
            return ExtractionResult(
                text="",
                metadata={"file": str(file_path)},
                extraction_method="pdf_error",
                errors=[str(e)]
            )
    
    async def _extract_with_pdfplumber(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto usando pdfplumber (mejor para tablas)"""
        
        all_text = []
        pages_data = []
        metadata = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata = {
                    "pages": len(pdf.pages),
                    "creator": pdf.metadata.get('Creator', ''),
                    "producer": pdf.metadata.get('Producer', ''),
                    "subject": pdf.metadata.get('Subject', ''),
                    "title": pdf.metadata.get('Title', ''),
                    "author": pdf.metadata.get('Author', '')
                }
                
                for i, page in enumerate(pdf.pages):
                    # Extraer texto
                    page_text = page.extract_text() or ""
                    
                    # Extraer tablas si están presentes
                    tables = page.extract_tables()
                    table_text = ""
                    if tables:
                        for table in tables:
                            table_rows = []
                            for row in table:
                                if row:
                                    table_rows.append("\t".join(str(cell or "") for cell in row))
                            table_text += "\n".join(table_rows) + "\n\n"
                    
                    combined_text = page_text
                    if table_text:
                        combined_text += "\n\nTABLAS:\n" + table_text
                    
                    all_text.append(combined_text)
                    
                    pages_data.append({
                        "page": i + 1,
                        "text_length": len(page_text),
                        "has_tables": len(tables) > 0,
                        "table_count": len(tables)
                    })
            
            combined_text = "\n\n".join(all_text)
            
            return ExtractionResult(
                text=combined_text,
                metadata=metadata,
                pages=pages_data,
                extraction_method="pdfplumber",
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"❌ Error con pdfplumber: {e}")
            raise
    
    async def _extract_with_pymupdf(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto usando PyMuPDF (mejor para texto general)"""
        
        all_text = []
        pages_data = []
        metadata = {}
        
        try:
            doc = fitz.open(file_path)
            
            # Metadata
            metadata = {
                "pages": doc.page_count,
                "title": doc.metadata.get('title', ''),
                "author": doc.metadata.get('author', ''),
                "subject": doc.metadata.get('subject', ''),
                "creator": doc.metadata.get('creator', ''),
                "producer": doc.metadata.get('producer', ''),
                "creation_date": doc.metadata.get('creationDate', ''),
                "modification_date": doc.metadata.get('modDate', '')
            }
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extraer texto
                page_text = page.get_text()
                all_text.append(page_text)
                
                pages_data.append({
                    "page": page_num + 1,
                    "text_length": len(page_text),
                    "rotation": page.rotation
                })
            
            doc.close()
            
            combined_text = "\n\n".join(all_text)
            
            return ExtractionResult(
                text=combined_text,
                metadata=metadata,
                pages=pages_data,
                extraction_method="pymupdf",
                confidence=0.90
            )
            
        except Exception as e:
            logger.error(f"❌ Error con PyMuPDF: {e}")
            raise
    
    async def _extract_with_pypdf2(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto usando PyPDF2 (método de respaldo)"""
        
        all_text = []
        pages_data = []
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                
                # Metadata
                info = reader.metadata
                if info:
                    metadata = {
                        "pages": len(reader.pages),
                        "title": info.get('/Title', ''),
                        "author": info.get('/Author', ''),
                        "subject": info.get('/Subject', ''),
                        "creator": info.get('/Creator', ''),
                        "producer": info.get('/Producer', '')
                    }
                else:
                    metadata = {"pages": len(reader.pages)}
                
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    all_text.append(page_text)
                    
                    pages_data.append({
                        "page": i + 1,
                        "text_length": len(page_text)
                    })
            
            combined_text = "\n\n".join(all_text)
            
            return ExtractionResult(
                text=combined_text,
                metadata=metadata,
                pages=pages_data,
                extraction_method="pypdf2",
                confidence=0.80
            )
            
        except Exception as e:
            logger.error(f"❌ Error con PyPDF2: {e}")
            raise
    
    async def _extract_with_fallback(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Intentar múltiples métodos automáticamente"""
        
        methods = ["pdfplumber", "pymupdf", "pypdf2"]
        last_error = None
        
        for method in methods:
            try:
                logger.info(f"🔄 Intentando método: {method}")
                result = await self.extract_text(file_path, {**options, "method": method})
                
                if result.text.strip():
                    logger.info(f"✅ Éxito con método: {method}")
                    return result
                    
            except Exception as e:
                logger.warning(f"⚠️ Falló método {method}: {e}")
                last_error = e
                continue
        
        # Si todos los métodos fallan
        return ExtractionResult(
            text="",
            metadata={"file": str(file_path)},
            extraction_method="all_methods_failed",
            errors=[str(last_error)] if last_error else ["Unknown error"]
        )
'''
        
        processors_dir = self.service_path / "processors"
        with open(processors_dir / "pdf_processor.py", "w", encoding="utf-8") as f:
            f.write(pdf_processor_content)
        
        self.logger.info("✅ Procesador de PDF creado")

    def create_ocr_service(self):
        """Crear servicio de OCR"""
        self.logger.info("👁️ Creando servicio de OCR...")
        
        ocr_service_content = '''"""
Servicio de OCR para extracción de texto de imágenes
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract
from services.text_extractor import ExtractionResult

logger = logging.getLogger(__name__)

class OCRService:
    """Servicio de OCR usando Tesseract"""
    
    def __init__(self):
        self.is_ready = False
        self.tesseract_available = False
        self.supported_languages = ['eng', 'spa']  # Inglés y Español por defecto
    
    async def initialize(self):
        """Inicializar el servicio de OCR"""
        logger.info("🔧 Inicializando OCR Service...")
        
        try:
            # Verificar si Tesseract está disponible
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("✅ Tesseract disponible")
            
            # Verificar idiomas disponibles
            available_langs = pytesseract.get_languages()
            self.supported_languages = [lang for lang in self.supported_languages if lang in available_langs]
            logger.info(f"📝 Idiomas disponibles: {self.supported_languages}")
            
        except Exception as e:
            logger.warning(f"⚠️ Tesseract no disponible: {e}")
            self.tesseract_available = False
        
        self.is_ready = True
        logger.info("✅ OCR Service inicializado")
    
    async def is_available(self) -> bool:
        """Verificar si OCR está disponible"""
        return self.tesseract_available
    
    async def extract_text(self, file_path: Path, options: Dict) -> ExtractionResult:
        """Extraer texto de imagen usando OCR"""
        
        if not self.tesseract_available:
            return ExtractionResult(
                text="",
                metadata={"file": str(file_path)},
                extraction_method="ocr_unavailable",
                errors=["Tesseract no está disponible"]
            )
        
        try:
            # Cargar imagen
            image = Image.open(file_path)
            
            # Preprocesar imagen si es necesario
            if options.get('preprocess', True):
                image = await self._preprocess_image(image, options)
            
            # Configurar OCR
            lang = options.get('language', 'eng+spa')
            config = options.get('config', '--psm 1 --oem 3')
            
            # Extraer texto
            text = pytesseract.image_to_string(
                image, 
                lang=lang, 
                config=config
            )
            
            # Obtener datos adicionales
            data = pytesseract.image_to_data(
                image, 
                lang=lang, 
                config=config, 
                output_type=pytesseract.Output.DICT
            )
            
            # Calcular confianza promedio
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100 if confidences else 0.0
            
            # Metadata
            metadata = {
                "image_size": image.size,
                "image_mode": image.mode,
                "language": lang,
                "config": config,
                "word_count": len(text.split()),
                "avg_confidence": avg_confidence,
                "total_words_detected": len([w for w in data['text'] if w.strip()]),
                "preprocessing_applied": options.get('preprocess', True)
            }
            
            return ExtractionResult(
                text=text.strip(),
                metadata=metadata,
                extraction_method="tesseract_ocr",
                confidence=avg_confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Error en OCR para {file_path.name}: {e}")
            return ExtractionResult(
                text="",
                metadata={"file": str(file_path)},
                extraction_method="ocr_error",
                errors=[str(e)]
            )
    
    async def _preprocess_image(self, image: Image.Image, options: Dict) -> Image.Image:
        """Preprocesar imagen para mejorar OCR"""
        
        try:
            # Convertir a array numpy
            img_array = np.array(image)
            
            # Convertir a escala de grises si no lo está
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Aplicar filtros según opciones
            if options.get('denoise', True):
                img_array = cv2.medianBlur(img_array, 3)
            
            if options.get('enhance_contrast', True):
                img_array = cv2.equalizeHist(img_array)
            
            if options.get('threshold', True):
                _, img_array = cv2.threshold(
                    img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            
            # Convertir de vuelta a PIL Image
            processed_image = Image.fromarray(img_array)
            
            # Redimensionar si es muy pequeña
            if processed_image.size[0] < 300 or processed_image.size[1] < 300:
                scale_factor = max(300 / processed_image.size[0], 300 / processed_image.size[1])
                new_size = (
                    int(processed_image.size[0] * scale_factor),
                    int(processed_image.size[1] * scale_factor)
                )
                processed_image = processed_image.resize(new_size, Image.Resampling.LANCZOS)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"⚠️ Error en preprocesamiento, usando imagen original: {e}")
            return image
    
    async def extract_from_pdf_pages(self, pdf_path: Path, options: Dict) -> List[ExtractionResult]:
        """Extraer texto de PDF usando OCR en cada página"""
        
        if not self.tesseract_available:
            return [ExtractionResult(
                text="",
                metadata={"file": str(pdf_path)},
                extraction_method="ocr_unavailable",
                errors=["Tesseract no está disponible"]
            )]
        
        try:
            from pdf2image import convert_from_path
            
            # Convertir PDF a imágenes
            pages = convert_from_path(pdf_path, dpi=300)
            
            results = []
            for i, page_image in enumerate(pages):
                logger.info(f"📄 Procesando página {i+1}/{len(pages)} con OCR")
                
                # Extraer texto de la página
                result = await self.extract_text_from_image(page_image, options)
                result.metadata.update({
                    "page_number": i + 1,
                    "total_pages": len(pages),
                    "source_file": str(pdf_path)
                })
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en OCR de PDF: {e}")
            return [ExtractionResult(
                text="",
                metadata={"file": str(pdf_path)},
                extraction_method="pdf_ocr_error",
                errors=[str(e)]
            )]
    
    async def extract_text_from_image(self, image: Image.Image, options: Dict) -> ExtractionResult:
        """Extraer texto directamente de una imagen PIL"""
        
        try:
            # Preprocesar imagen
            if options.get('preprocess', True):
                image = await self._preprocess_image(image, options)
            
            # Configurar OCR
            lang = options.get('language', 'eng+spa')
            config = options.get('config', '--psm 1 --oem 3')
            
            # Extraer texto
            text = pytesseract.image_to_string(
                image, 
                lang=lang, 
                config=config
            )
            
            # Obtener datos de confianza
            data = pytesseract.image_to_data(
                image, 
                lang=lang, 
                config=config, 
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100 if confidences else 0.0
            
            return ExtractionResult(
                text=text.strip(),
                metadata={
                    "image_size": image.size,
                    "language": lang,
                    "avg_confidence": avg_confidence,
                    "word_count": len(text.split())
                },
                extraction_method="tesseract_ocr",
                confidence=avg_confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Error en OCR de imagen: {e}")
            return ExtractionResult(
                text="",
                metadata={},
                extraction_method="image_ocr_error",
                errors=[str(e)]
            )
    
    def get_available_languages(self) -> List[str]:
        """Obtener idiomas disponibles para OCR"""
        try:
            return pytesseract.get_languages() if self.tesseract_available else []
        except:
            return []
'''
        
        services_dir = self.service_path / "services"
        with open(services_dir / "ocr_service.py", "w", encoding="utf-8") as f:
            f.write(ocr_service_content)
        
        self.logger.info("✅ Servicio de OCR creado")

    def create_routers(self):
        """Crear routers de FastAPI"""
        self.logger.info("🌐 Creando routers...")
        
        # Router de upload
        upload_router = '''from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import shutil
import uuid
from pathlib import Path
import logging

from services.document_manager import DocumentManager
from services.text_extractor import TextExtractor
from utils.file_utils import validate_file, get_file_info

router = APIRouter()
logger = logging.getLogger(__name__)

# Configuración de archivos
UPLOAD_DIR = Path("data/uploads")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.tiff', '.xlsx', '.csv'}

@router.post("/file")
async def upload_file(
    file: UploadFile = File(...),
    extract_text: bool = Form(True),
    ocr_enabled: bool = Form(True),
    background_tasks: BackgroundTasks = None
):
    """Subir y procesar un archivo"""
    
    try:
        # Validar archivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo requerido")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Formato no soportado. Permitidos: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Generar ID único para el archivo
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        # Crear directorio si no existe
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validar tamaño
        if file_path.stat().st_size > MAX_FILE_SIZE:
            file_path.unlink()  # Eliminar archivo
            raise HTTPException(
                status_code=413, 
                detail=f"Archivo muy grande. Máximo: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Obtener información del archivo
        file_info = await get_file_info(file_path)
        
        response_data = {
            "file_id": file_id,
            "filename": file.filename,
            "size": file_info.get("size", 0),
            "format": file_ext,
            "upload_status": "success",
            "file_path": str(file_path)
        }
        
        # Extraer texto si se solicita
        if extract_text:
            if background_tasks:
                # Procesamiento en background
                background_tasks.add_task(
                    process_file_background, 
                    file_path, 
                    file_id, 
                    {"ocr_enabled": ocr_enabled}
                )
                response_data["processing"] = "background"
            else:
                # Procesamiento inmediato
                try:
                    extractor = TextExtractor()
                    result = await extractor.extract_text(
                        file_path, 
                        extraction_options={"ocr_enabled": ocr_enabled}
                    )
                    response_data.update({
                        "text": result.text[:1000] + "..." if len(result.text) > 1000 else result.text,
                        "text_length": len(result.text),
                        "extraction_method": result.extraction_method,
                        "confidence": result.confidence,
                        "processing": "immediate"
                    })
                except Exception as e:
                    logger.error(f"Error extrayendo texto: {e}")
                    response_data["extraction_error"] = str(e)
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error subiendo archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def upload_batch(
    files: List[UploadFile] = File(...),
    extract_text: bool = Form(True),
    background_tasks: BackgroundTasks = None
):
    """Subir múltiples archivos en lote"""
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Máximo 10 archivos por lote")
    
    results = []
    
    for file in files:
        try:
            result = await upload_file(file, extract_text, background_tasks=background_tasks)
            results.append({"filename": file.filename, "result": result.body})
        except Exception as e:
            results.append({
                "filename": file.filename, 
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "batch_id": str(uuid.uuid4()),
        "total_files": len(files),
        "results": results
    }

async def process_file_background(file_path: Path, file_id: str, options: dict):
    """Procesar archivo en background"""
    try:
        extractor = TextExtractor()
        result = await extractor.extract_text(file_path, extraction_options=options)
        
        # Aquí guardarías el resultado en base de datos o cache
        logger.info(f"✅ Archivo {file_id} procesado en background")
        
    except Exception as e:
        logger.error(f"❌ Error procesando archivo {file_id} en background: {e}")
'''
        
        routers_dir = self.service_path / "routers"
        with open(routers_dir / "upload.py", "w", encoding="utf-8") as f:
            f.write(upload_router)
        
        # Router de extracción
        extraction_router = '''from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from services.text_extractor import TextExtractor
from services.ocr_service import OCRService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/text")
async def extract_text(
    file_path: str,
    format_hint: Optional[str] = None,
    method: Optional[str] = None,
    language: str = "eng+spa",
    preprocess: bool = True
):
    """Extraer texto de un archivo"""
    
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        extractor = TextExtractor()
        
        options = {
            "method": method,
            "language": language,
            "preprocess": preprocess
        }
        
        result = await extractor.extract_text(
            file_path_obj, 
            format_hint=format_hint,
            extraction_options=options
        )
        
        return {
            "text": result.text,
            "metadata": result.metadata,
            "extraction_method": result.extraction_method,
            "confidence": result.confidence,
            "pages": result.pages,
            "errors": result.errors
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except Exception as e:
        logger.error(f"Error extrayendo texto: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ocr")
async def extract_with_ocr(
    file_path: str,
    language: str = "eng+spa",
    preprocess: bool = True,
    enhance_contrast: bool = True,
    denoise: bool = True
):
    """Extraer texto usando OCR"""
    
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        ocr_service = OCRService()
        
        options = {
            "language": language,
            "preprocess": preprocess,
            "enhance_contrast": enhance_contrast,
            "denoise": denoise
        }
        
        result = await ocr_service.extract_text(file_path_obj, options)
        
        return {
            "text": result.text,
            "metadata": result.metadata,
            "confidence": result.confidence,
            "errors": result.errors
        }
        
    except Exception as e:
        logger.error(f"Error en OCR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/methods")
async def available_methods():
    """Obtener métodos de extracción disponibles"""
    
    ocr_service = OCRService()
    
    return {
        "pdf_methods": ["pdfplumber", "pymupdf", "pypdf2", "auto"],
        "ocr_available": await ocr_service.is_available(),
        "ocr_languages": ocr_service.get_available_languages(),
        "supported_formats": [
            "pdf", "docx", "txt", "rtf", 
            "png", "jpg", "jpeg", "tiff", "bmp",
            "xlsx", "xls", "csv"
        ]
    }
'''
        
        with open(routers_dir / "extraction.py", "w", encoding="utf-8") as f:
            f.write(extraction_router)
        
        # Crear otros routers básicos
        for router_name in ["parsing", "conversion", "analysis"]:
            basic_router = f'''from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

@router.get("/")
async def {router_name}_root():
    return {{"message": "{router_name.title()} router - En desarrollo"}}
'''
            with open(routers_dir / f"{router_name}.py", "w", encoding="utf-8") as f:
                f.write(basic_router)
        
        self.logger.info("✅ Routers creados")

    def create_config_utils(self):
        """Crear utilidades de configuración"""
        self.logger.info("⚙️ Creando utilidades...")
        
        # Config
        config_content = '''"""
Configuración del Document-Processor
"""
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Configuración general
    app_name: str = "Document-Processor"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8002
    
    # Archivos y directorios
    upload_dir: str = "./data/uploads"
    processed_dir: str = "./data/processed"
    cache_dir: str = "./data/cache"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    # Formatos soportados
    allowed_extensions: List[str] = [
        ".pdf", ".docx", ".txt", ".rtf",
        ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
        ".xlsx", ".xls", ".csv"
    ]
    
    # OCR
    ocr_enabled: bool = True
    ocr_languages: str = "eng+spa"
    ocr_dpi: int = 300
    
    # PDF
    pdf_method: str = "pdfplumber"  # pdfplumber, pymupdf, pypdf2
    
    # CORS
    cors_origins: List[str] = ["http://localhost:8080", "http://127.0.0.1:8080"]
    
    # Base de datos
    database_url: str = "sqlite:///./document_processor.db"
    redis_url: str = "redis://localhost:6379/2"
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    
    # Tesseract (si no está en PATH)
    tesseract_cmd: str = ""  # Dejar vacío para autodetección
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
'''
        
        utils_dir = self.service_path / "utils"
        with open(utils_dir / "config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        # File utils
        file_utils_content = '''"""
Utilidades para manejo de archivos
"""
import os
import magic
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional

async def detect_file_type(file_path: Path) -> Dict[str, str]:
    """Detectar tipo de archivo usando múltiples métodos"""
    
    result = {
        "extension": file_path.suffix.lower(),
        "mime_type": "",
        "magic_type": "",
        "detected_format": ""
    }
    
    try:
        # Usando python-magic
        result["magic_type"] = magic.from_file(str(file_path))
        result["mime_type"] = magic.from_file(str(file_path), mime=True)
    except:
        # Fallback a mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        result["mime_type"] = mime_type or "application/octet-stream"
    
    # Determinar formato basado en extensión y mime type
    ext = result["extension"]
    mime = result["mime_type"]
    
    if ext in ['.pdf'] or 'pdf' in mime:
        result["detected_format"] = "pdf"
    elif ext in ['.docx'] or 'wordprocessingml' in mime:
        result["detected_format"] = "docx"
    elif ext in ['.doc'] or 'msword' in mime:
        result["detected_format"] = "doc"
    elif ext in ['.txt'] or 'text/plain' in mime:
        result["detected_format"] = "txt"
    elif ext in ['.rtf'] or 'rtf' in mime:
        result["detected_format"] = "rtf"
    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'] or 'image' in mime:
        result["detected_format"] = "image"
    elif ext in ['.xlsx'] or 'spreadsheetml' in mime:
        result["detected_format"] = "xlsx"
    elif ext in ['.xls'] or 'excel' in mime:
        result["detected_format"] = "xls"
    elif ext in ['.csv'] or 'csv' in mime:
        result["detected_format"] = "csv"
    else:
        result["detected_format"] = "unknown"
    
    return result

async def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Obtener información completa del archivo"""
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    stat = file_path.stat()
    file_type_info = await detect_file_type(file_path)
    
    return {
        "name": file_path.name,
        "size": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "extension": file_path.suffix.lower(),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "mime_type": file_type_info["mime_type"],
        "detected_format": file_type_info["detected_format"],
        "path": str(file_path)
    }

async def validate_file(file_path: Path, max_size: int = None, allowed_extensions: list = None) -> Dict[str, Any]:
    """Validar archivo según criterios"""
    
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Verificar existencia
    if not file_path.exists():
        validation_result["valid"] = False
        validation_result["errors"].append("Archivo no encontrado")
        return validation_result
    
    # Verificar tamaño
    if max_size:
        size = file_path.stat().st_size
        if size > max_size:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Archivo muy grande: {size} bytes (máximo: {max_size} bytes)"
            )
    
    # Verificar extensión
    if allowed_extensions:
        ext = file_path.suffix.lower()
        if ext not in allowed_extensions:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Extensión no permitida: {ext} (permitidas: {allowed_extensions})"
            )
    
    # Verificar si es archivo válido (no directorio)
    if file_path.is_dir():
        validation_result["valid"] = False
        validation_result["errors"].append("Es un directorio, no un archivo")
    
    return validation_result
'''
        with open(utils_dir / "file_utils.py", "w", encoding="utf-8") as f:
            f.write(file_utils_content)
        
        self.logger.info("✅ Utilidades creadas")

    def run(self):
        """Ejecutar todos los pasos de la configuración"""
        if not self.validate_environment():
            return
        
        self.create_directory_structure()
        self.create_main_app()
        self.create_requirements()
        self.create_text_extractor()
        self.create_pdf_processor()
        self.create_ocr_service()
        self.create_routers()
        self.create_config_utils()
        
        self.logger.info("🎉 ¡Configuración del Document-Processor completada!")

if __name__ == "__main__":
    setup = DocumentProcessorSetup()
    setup.run()