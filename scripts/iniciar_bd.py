#!/usr/bin/env python3
"""
Script para inicializar la base de datos.
Este script crea la base de datos inicial y ejecuta las migraciones pendientes.
"""
import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path de Python
sys.path.insert(0, str(Path(__file__).parent.parent))

def inicializar_bd():
    """Inicializa la base de datos y ejecuta migraciones."""
    from gateway.database import init_db, engine
    from gateway.models import Base
    
    print("🔧 Inicializando base de datos...")
    
    # Crear todas las tablas
    Base.metadata.create_all(bind=engine)
    
    print("✅ ¡Base de datos inicializada exitosamente!")
    print("📝 Puedes crear una migración con: alembic revision --autogenerate -m 'descripcion'")
    print("🚀 Luego aplica las migraciones con: alembic upgrade head")

if __name__ == "__main__":
    inicializar_bd()
