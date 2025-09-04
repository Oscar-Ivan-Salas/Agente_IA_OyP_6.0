"""
Gestor de modelos
"""

import logging

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.engine = None
    
    def set_engine(self, engine):
        """Configurar motor"""
        self.engine = engine
        logger.info("✅ Motor configurado")
    
    async def reload_models(self):
        """Recargar modelos"""
        if self.engine:
            await self.engine.initialize()
            return True
        return False
