#!/usr/bin/env python3
"""
Pruebas del AI Engine v6.0
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_functionality():
    """Probar funcionalidad"""
    print("🧪 PROBANDO AI ENGINE v6.0")
    print("=" * 40)
    
    try:
        # Test 1: Importar motor
        print("\n1️⃣ Importando motor...")
        from models.llm_hybrid_engine import HybridLLMEngine
        print("✅ Motor importado")
        
        # Test 2: Inicializar
        print("\n2️⃣ Inicializando...")
        engine = HybridLLMEngine()
        await engine.initialize()
        print("✅ Motor inicializado")
        
        # Test 3: Estado
        print("\n3️⃣ Verificando estado...")
        is_ready = engine.is_ready()
        print(f"✅ Listo: {is_ready}")
        
        # Test 4: Health check
        print("\n4️⃣ Health check...")
        health = await engine.health_check()
        print(f"✅ Estado: {health['status']}")
        
        # Test 5: Modelos
        print("\n5️⃣ Modelos disponibles...")
        models = await engine.get_available_models()
        print(f"✅ Locales: {len(models['local'])}")
        print(f"✅ Cloud: {len(models['cloud'])}")
        
        # Test 6: Generación
        print("\n6️⃣ Probando generación...")
        result = await engine.generate_text("Hola mundo")
        print(f"✅ Respuesta: {result['response'][:50]}...")
        print(f"✅ Motor: {result['engine']}")
        
        # Test 7: Análisis
        print("\n7️⃣ Probando análisis...")
        sentiment = await engine.analyze_sentiment("Me gusta esto")
        print(f"✅ Sentimiento: {sentiment['sentiment']}")
        
        # Test 8: Estadísticas
        print("\n8️⃣ Estadísticas...")
        stats = await engine.get_stats()
        print(f"✅ Requests: {stats['requests_total']}")
        print(f"✅ Success rate: {stats['success_rate']:.1f}%")
        
        # Cleanup
        await engine.cleanup()
        
        print("\n" + "=" * 40)
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("=" * 40)
        print("\n🚀 Siguiente paso:")
        print("   uvicorn app:app --host 0.0.0.0 --port 8001")
        print("\n🌐 URLs:")
        print("   • API: http://localhost:8001")
        print("   • Docs: http://localhost:8001/docs")
        print("   • Health: http://localhost:8001/health")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_functionality())
    sys.exit(0 if success else 1)
