"""
Configuración para las pruebas unitarias del gateway
"""
import pytest
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Configuración de pruebas
@pytest.fixture
def sample_config():
    """Configuración de ejemplo para pruebas"""
    return {
        "ai_engine": {
            "url": "http://test-ai-engine:8000",
            "health_endpoint": "/health",
            "name": "AI Engine",
            "description": "Motor de IA"
        },
        "document_processor": {
            "url": "http://test-document-processor:8000",
            "health_endpoint": "/health",
            "name": "Document Processor",
            "description": "Procesador de documentos"
        }
    }
