#!/usr/bin/env python3
"""
Test de importaciones del proyecto
"""

try:
    from fastapi import FastAPI
    print("✅ FastAPI importado correctamente")
except ImportError as e:
    print(f"❌ Error importando FastAPI: {e}")

try:
    from routes import dashboard
    print("✅ Módulo dashboard importado correctamente")
except ImportError as e:
    print(f"❌ Error importando dashboard: {e}")
    
    # Mostrar el path de Python
    import sys
    print("\nPython path:")
    for p in sys.path:
        print(f" - {p}")
    
    # Mostrar el directorio actual
    import os
    print(f"\nDirectorio actual: {os.getcwd()}")
