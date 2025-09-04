#!/usr/bin/env python3
"""
Script de prueba para verificar las importaciones del Gateway.
"""
import sys
import os

# Asegurarse de que el directorio del proyecto esté en el path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=== Prueba de importaciones del Gateway ===")
print(f"Directorio del proyecto: {project_root}")
print(f"Python path: {sys.path}")

try:
    print("\nIntentando importar gateway.app...")
    from gateway.app import app
    print("✅ gateway.app importado correctamente!")
    
    print("\nAtributos de la aplicación:")
    print(f"- Título: {app.title}")
    print(f"- Descripción: {app.description}")
    print(f"- Versión: {app.version}")
    
    print("\n✅ ¡Todas las importaciones son correctas!")
    print("Puedes iniciar el servidor con: python -m uvicorn gateway.app:app --reload")
    
except ImportError as e:
    print(f"\n❌ Error de importación: {e}")
    print("\nTraceback completo:")
    import traceback
    traceback.print_exc()
    
    print("\nSugerencias:")
    print("1. Verifica que el directorio del proyecto esté en el PYTHONPATH")
    print("2. Asegúrate de que todos los paquetes requeridos estén instalados")
    print("3. Revisa que el archivo gateway/__init__.py exista y esté vacío")
