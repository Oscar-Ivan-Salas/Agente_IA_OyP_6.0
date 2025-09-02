""
Script para ejecutar pruebas manuales del ProxyManager
"""
import asyncio
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

# Importar el proxy_manager después de configurar el path
try:
    from gateway.src.proxy_manager import ProxyManager, CircuitBreakerState
    print("✅ Módulos importados correctamente")
    
    # Configuración de prueba
    test_config = {
        "ai_engine": {
            "url": "http://test-ai-engine:8000",
            "health_endpoint": "/health",
            "name": "AI Engine",
            "description": "Motor de IA"
        }
    }
    
    # Crear instancia
    pm = ProxyManager()
    pm.services = test_config
    pm._init_circuit_breakers()
    pm._init_health_cache()
    
    # Probar inicialización
    print("\n🔍 Probando inicialización...")
    print(f"Servicios configurados: {list(pm.services.keys())}")
    print(f"Estado del circuit breaker: {pm.circuit_breakers['ai_engine']['state']}")
    
    print("\n✅ Prueba completada exitosamente!")
    
except Exception as e:
    print(f"\n❌ Error durante la prueba: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
