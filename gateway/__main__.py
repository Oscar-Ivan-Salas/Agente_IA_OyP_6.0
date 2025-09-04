#!/usr/bin/env python3
"""
Punto de entrada principal para el Gateway.
"""
import sys
import os

# Asegurarse de que el directorio del proyecto esté en el path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configurar logging básico
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    try:
        from gateway.app import app
        import uvicorn
        
        print("🚀 Iniciando Gateway...")
        print(f"Directorio del proyecto: {project_root}")
        
        uvicorn.run(
            "gateway.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Error al iniciar el servidor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
