"""
Pruebas unitarias para el ProxyManager del Gateway
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException
import httpx
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Importar el proxy_manager para pruebas
from gateway.src.proxy_manager import (
    ProxyManager,
    CircuitBreakerState,
    ServiceHealthStatus
)

# Configuración de prueba para los servicios
TEST_CONFIG = {
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

# Fixture para el proxy_manager
@pytest.fixture
def proxy_manager():
    """Fixture que devuelve una instancia de ProxyManager para pruebas"""
    # Crear instancia con configuración de prueba
    pm = ProxyManager()
    pm.services = TEST_CONFIG
    
    # Inicializar circuit breakers
    pm._init_circuit_breakers()
    pm._init_health_cache()
    
    yield pm
    
    # Limpieza
    asyncio.run(pm.close())

# Tests para la funcionalidad básica
class TestProxyManagerBasics:
    """Pruebas para la funcionalidad básica del ProxyManager"""
    
    def test_initialization(self, proxy_manager):
        """Verificar que el ProxyManager se inicializa correctamente"""
        assert isinstance(proxy_manager, ProxyManager)
        assert hasattr(proxy_manager, 'services')
        assert 'ai_engine' in proxy_manager.services
        assert 'document_processor' in proxy_manager.services
        
    def test_circuit_breaker_initial_state(self, proxy_manager):
        """Verificar que los circuit breakers se inicializan en estado cerrado"""
        for service_id in proxy_manager.services.keys():
            assert proxy_manager.circuit_breakers[service_id]['state'] == CircuitBreakerState.CLOSED
            assert proxy_manager.circuit_breakers[service_id]['failure_count'] == 0

# Tests para el manejo de peticiones
class TestRequestHandling:
    """Pruebas para el manejo de peticiones HTTP"""
    
    @pytest.mark.asyncio
    async def test_successful_request(self, proxy_manager):
        """Verificar que una petición exitosa se maneja correctamente"""
        # Mock de la respuesta exitosa
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        
        # Mock del cliente HTTP
        with patch('httpx.AsyncClient.request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            # Realizar petición
            result = await proxy_manager.proxy_request(
                service_id="ai_engine",
                method="GET",
                path="/test"
            )
            
            # Verificar resultados
            assert result['status_code'] == 200
            assert result['data'] == {"status": "ok"}
            
            # Verificar que el circuit breaker sigue cerrado
            assert proxy_manager.circuit_breakers["ai_engine"]['state'] == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_failed_request(self, proxy_manager):
        """Verificar que un error en la petición se maneja correctamente"""
        # Mock de la respuesta fallida
        with patch('httpx.AsyncClient.request', side_effect=httpx.ConnectError("Connection error")):
            # Verificar que se lanza la excepción HTTP esperada
            with pytest.raises(HTTPException) as exc_info:
                await proxy_manager.proxy_request(
                    service_id="ai_engine",
                    method="GET",
                    path="/test"
                )
            
            # Verificar que se lanza el código de estado correcto
            assert exc_info.value.status_code == 503
            
            # Verificar que el contador de fallos se incrementó
            assert proxy_manager.circuit_breakers["ai_engine"]['failure_count'] == 1

# Tests para el circuit breaker
class TestCircuitBreaker:
    """Pruebas para el funcionamiento del circuit breaker"""
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, proxy_manager):
        """Verificar que el circuit breaker se abre después de varios fallos"""
        failure_threshold = proxy_manager.circuit_breakers["ai_engine"]['failure_threshold']
        
        # Simular fallos hasta alcanzar el umbral
        with patch('httpx.AsyncClient.request', side_effect=httpx.ConnectError("Connection error")):
            for _ in range(failure_threshold):
                with pytest.raises(HTTPException):
                    await proxy_manager.proxy_request(
                        service_id="ai_engine",
                        method="GET",
                        path="/test"
                    )
        
        # Verificar que el circuit breaker está abierto
        assert proxy_manager.circuit_breakers["ai_engine"]['state'] == CircuitBreakerState.OPEN
        
        # Verificar que las nuevas peticiones fallan inmediatamente
        with pytest.raises(HTTPException) as exc_info:
            await proxy_manager.proxy_request(
                service_id="ai_engine",
                method="GET",
                path="/test"
            )
        
        assert "temporalmente no disponible" in str(exc_info.value.detail)
        assert exc_info.value.status_code == 503

# Tests para health checks
class TestHealthChecks:
    """Pruebas para los health checks de los servicios"""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, proxy_manager):
        """Verificar que un health check exitoso actualiza el estado del servicio"""
        # Mock de respuesta exitosa
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        
        with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            result = await proxy_manager.health_check("ai_engine")
            
            # Verificar resultados
            assert result["status"] == ServiceHealthStatus.HEALTHY
            assert "response_time_ms" in result
            
            # Verificar que se actualizó el caché de salud
            assert proxy_manager.service_health["ai_engine"]["status"] == ServiceHealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, proxy_manager):
        """Verificar que un health check fallido actualiza el estado correctamente"""
        # Mock de respuesta fallida
        with patch('httpx.AsyncClient.get', side_effect=httpx.ConnectError("Connection error")):
            result = await proxy_manager.health_check("ai_engine")
            
            # Verificar resultados
            assert result["status"] == ServiceHealthStatus.UNHEALTHY
            assert "error" in result
            
            # Verificar que se actualizó el caché de salud
            assert proxy_manager.service_health["ai_engine"]["status"] == ServiceHealthStatus.UNHEALTHY
            assert proxy_manager.service_health["ai_engine"]["consecutive_failures"] == 1

# Tests para el manejo de métricas
class TestMetrics:
    """Pruebas para el seguimiento de métricas"""
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, proxy_manager):
        """Verificar que se registran correctamente las métricas de las peticiones"""
        # Mock de respuesta exitosa
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        
        with patch('httpx.AsyncClient.request', return_value=mock_response):
            # Realizar varias peticiones
            for _ in range(3):
                await proxy_manager.proxy_request(
                    service_id="ai_engine",
                    method="GET",
                    path="/test"
                )
        
        # Verificar métricas
        metrics = proxy_manager.get_metrics()
        ai_metrics = metrics["services_metrics"]["ai_engine"]
        
        assert ai_metrics["total_requests"] == 3
        assert ai_metrics["successful_requests"] == 3
        assert ai_metrics["failed_requests"] == 0
        assert ai_metrics["avg_response_time"] > 0
        assert ai_metrics["last_request"] is not None

# Tests para el cierre del cliente
class TestShutdown:
    """Pruebas para el cierre correcto del cliente HTTP"""
    
    @pytest.mark.asyncio
    async def test_client_shutdown(self, proxy_manager):
        """Verificar que el cliente HTTP se cierra correctamente"""
        # Mock del método aclose
        with patch.object(proxy_manager.client, 'aclose') as mock_aclose:
            await proxy_manager.close()
            mock_aclose.assert_called_once()
