#!/usr/bin/env python3
"""
Script principal para iniciar el sistema Agente IA OYP 6.0
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('startup.log')
    ]
)
logger = logging.getLogger(__name__)

def check_venv():
    """Verificar si estamos en un entorno virtual"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def install_dependencies():
    """Instalar dependencias si es necesario"""
    logger.info("Verificando dependencias...")
    try:
        import uvicorn
        import fastapi
        logger.info("Todas las dependencias están instaladas")
    except ImportError:
        logger.info("Instalando dependencias...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def start_gateway():
    """Iniciar el servidor gateway"""
    logger.info("Iniciando servidor gateway...")
    try:
        import uvicorn
        from gateway.app import app
        
        uvicorn.run(
            "gateway.app:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error al iniciar el gateway: {e}")
        return False
    return True

def ensure_static_dirs():
    """Asegura que los directorios estáticos existan"""
    static_dirs = [
        os.path.join("gateway", "static"),
        os.path.join("gateway", "templates")
    ]
    
    for dir_path in static_dirs:
        os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(os.path.join(dir_path, ".gitkeep")):
            open(os.path.join(dir_path, ".gitkeep"), 'a').close()

def setup_logging():
    """Configura el logging para manejar caracteres Unicode en Windows"""
    import sys
    
    class UnicodeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                if sys.stdout.encoding != 'utf-8':
                    msg = msg.encode('utf-8', 'replace').decode('ascii', 'replace')
                print(msg, file=sys.stderr)
            except Exception:
                self.handleError(record)
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            UnicodeStreamHandler(),
            logging.FileHandler('startup.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Función principal"""
    global logger
    logger = setup_logging()
    
    logger.info("=" * 50)
    logger.info("Iniciando Agente IA OYP 6.0")
    logger.info(f"Directorio actual: {os.getcwd()}")
    
    # Asegurar que los directorios necesarios existan
    ensure_static_dirs()
    
    # Verificar entorno virtual
    if not check_venv():
        logger.warning("ADVERTENCIA: No estás en un entorno virtual. Se recomienda activar el entorno virtual primero.")
        logger.info("Puedes activarlo con: source venv/bin/activate (Linux/Mac) o venv\\Scripts\\activate (Windows)")
    
    # Instalar dependencias
    install_dependencies()
    
    # Iniciar gateway
    logger.info("Sistema listo")
    logger.info("Abre tu navegador en: http://localhost:8080")
    logger.info("Presiona Ctrl+C para detener el servidor")
    
    start_gateway()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n¡Hasta luego! 👋")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)
