"""
Script para inicializar la base de datos del Gateway.
"""
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para que Python encuentre los módulos
sys.path.append(str(Path(__file__).parent))

# Importar los modelos primero para que se registren con SQLAlchemy
from gateway.models import *
from gateway.database import engine, Base

def main():
    print("Inicializando la base de datos...")
    
    try:
        # Crear todas las tablas
        Base.metadata.create_all(bind=engine)
        print("✅ Base de datos inicializada correctamente.")
    except Exception as e:
        print(f"❌ Error al inicializar la base de datos: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
