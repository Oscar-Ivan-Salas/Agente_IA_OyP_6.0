#!/usr/bin/env python3
"""
Script para iniciar el servidor Gateway.
"""
import sys
import os
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gateway_debug.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Asegurarse de que el directorio del proyecto esté en el path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        logger.info(f"Directorio del proyecto: {project_root}")
        logger.info(f"Python path: {sys.path}")
        
        # Verificar que podemos importar la aplicación
        try:
            from gateway.app import app
            logger.info("✅ Módulo gateway importado correctamente")
        except ImportError as e:
            logger.error(f"❌ Error al importar el módulo gateway: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1
            
        # Iniciar el servidor
        import uvicorn
        logger.info("🚀 Iniciando servidor Uvicorn...")
        uvicorn.run(
            "gateway.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
