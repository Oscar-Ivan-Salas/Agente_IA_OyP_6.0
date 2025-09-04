#!/usr/bin/env python3
"""
Script para iniciar el Gateway.
"""
import sys
import os

# Asegurarse de que el directorio del proyecto esté en el path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Verificar que podemos importar el módulo gateway
try:
    from gateway.app import app
    print("✅ Módulo gateway importado correctamente")
except Exception as e:
    print(f"❌ Error al importar el módulo gateway: {e}")
    print(f"Python path: {sys.path}")
    raise

# Iniciar el servidor
if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor Gateway...")
    uvicorn.run(
        "gateway.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
