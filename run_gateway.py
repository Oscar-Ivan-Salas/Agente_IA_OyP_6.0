#!/usr/bin/env python3
"""
Script para iniciar el Gateway.
"""
import sys
import os
import uvicorn

# Asegurarse de que el directorio del proyecto esté en el path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configuración del servidor
config = {
    "app": "gateway.app:app",
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info"
}

if __name__ == "__main__":
    print("🚀 Iniciando Gateway...")
    print(f"Directorio de trabajo: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    try:
        # Intentar importar la aplicación
        from gateway.app import app
        print("✅ Módulo gateway importado correctamente")
        
        # Iniciar el servidor
        uvicorn.run(**config)
    except Exception as e:
        print(f"❌ Error al iniciar el servidor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
