"""
Configuración global para las pruebas del Agente IA OyP 6.0
"""
import os
import sys
import asyncio
import pytest
from pathlib import Path

# Agregar el directorio raíz al path para importar los módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuración de asyncio para Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Fixture para el event loop
@pytest.fixture(scope="session")
def event_loop():
    """
    Crea una instancia del event loop para la sesión de pruebas.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

# Configuración de variables de entorno para pruebas
@pytest.fixture(autouse=True, scope="session")
def setup_environment():
    """
    Configura las variables de entorno para las pruebas.
    """
    # Configuración de base de datos en memoria para pruebas
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DEBUG"] = "True"
    
    # Configuración de servicios de prueba
    os.environ["AI_ENGINE_URL"] = "http://test-ai-engine:8000"
    os.environ["DOCUMENT_PROCESSOR_URL"] = "http://test-document-processor:8000"
    os.environ["ANALYTICS_ENGINE_URL"] = "http://test-analytics-engine:8000"
    os.environ["REPORT_GENERATOR_URL"] = "http://test-report-generator:8000"
    os.environ["CHAT_AI_SERVICE_URL"] = "http://test-chat-ai-service:8000"

# Fixture para el directorio temporal de pruebas
@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """
    Crea un directorio temporal para las pruebas.
    """
    return tmp_path_factory.mktemp("test_data")

# Fixture para limpiar imports después de las pruebas
@pytest.fixture(autouse=True)
def cleanup_imports():
    """
    Limpia los imports después de cada prueba.
    """
    # Guardar módulos importados antes de la prueba
    before = set(sys.modules.keys())
    
    yield
    
    # Limpiar módulos importados durante la prueba
    after = set(sys.modules.keys())
    for module in (after - before):
        if module.startswith('gateway'):
            del sys.modules[module]
