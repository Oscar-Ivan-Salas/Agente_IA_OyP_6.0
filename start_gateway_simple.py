#!/usr/bin/env python3
"""
Script simplificado para iniciar el Gateway.
"""
import sys
import os

# Configurar el path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configurar logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=== Iniciando Gateway ===")
        logger.info(f"Directorio del proyecto: {project_root}")
        logger.info(f"Python path: {sys.path}")
        
        # Intentar importar la aplicación
        try:
            logger.info("Intentando importar gateway.app...")
            from gateway.app import app
            logger.info("✅ gateway.app importado correctamente")
            
            # Iniciar el servidor
            import uvicorn
            logger.info("🚀 Iniciando servidor Uvicorn...")
            uvicorn.run(
                "gateway.app:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
                log_level="debug"
            )
            
        except ImportError as e:
            logger.error(f"❌ Error al importar gateway.app: {e}")
            logger.error("\n=== Traceback completo ===")
            import traceback
            logger.error(traceback.format_exc())
            
            # Verificar si el archivo existe
            app_path = os.path.join(project_root, 'gateway', 'app.py')
            if not os.path.exists(app_path):
                logger.error(f"\n❌ El archivo {app_path} no existe")
            else:
                logger.info(f"\n✅ El archivo {app_path} existe")
                try:
                    with open(app_path, 'r', encoding='utf-8') as f:
                        first_lines = [next(f) for _ in range(10)]
                    logger.info("\nPrimeras líneas del archivo app.py:")
                    for line in first_lines:
                        logger.info(line.strip())
                except Exception as e:
                    logger.error(f"Error al leer el archivo app.py: {e}")
            
            return 1
            
    except Exception as e:
        logger.error(f"\n❌ Error inesperado: {e}")
        logger.error("\n=== Traceback completo ===")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
