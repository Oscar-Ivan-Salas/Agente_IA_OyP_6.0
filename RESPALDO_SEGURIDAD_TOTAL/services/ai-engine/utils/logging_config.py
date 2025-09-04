"""
Configuración de logging
"""

import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup logging"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('./logs/ai_engine.log')
        ]
    )
