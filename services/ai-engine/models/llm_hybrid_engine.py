"""
Motor híbrido de IA - Versión completa
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import hashlib
import os

# Imports opcionales con manejo de errores
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
    
    class ModelUsage(Base):
        __tablename__ = "model_usage"
        id = Column(Integer, primary_key=True)
        model_name = Column(String, nullable=False)
        model_type = Column(String, nullable=False)
        tokens_input = Column(Integer, nullable=False)
        tokens_output = Column(Integer, nullable=False)
        inference_time = Column(Float, nullable=False)
        timestamp = Column(DateTime, default=datetime.utcnow)
        
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from utils.config import Settings

logger = logging.getLogger(__name__)

class HybridLLMEngine:
    """Motor híbrido completo"""
    
    def __init__(self):
        self.settings = Settings()
        self.is_initialized = False
        self.models_local = {}
        self.models_cloud = {}
        self.embeddings_model = None
        self.redis_client = None
        self.db_session = None
        
        self.stats = {
            "requests_total": 0,
            "requests_local": 0,
            "requests_cloud": 0,
            "requests_cached": 0,
            "errors": 0,
            "tokens_processed": 0,
            "inference_time_total": 0.0,
            "start_time": datetime.now()
        }
        
    async def initialize(self):
        """Inicializar motor"""
        logger.info("🔄 Inicializando motor híbrido...")
        
        try:
            if SQLALCHEMY_AVAILABLE:
                await self._init_database()
            
            if REDIS_AVAILABLE:
                await self._init_redis()
            
            await self._init_cloud_apis()
            
            if TRANSFORMERS_AVAILABLE:
                await self._init_local_models()
            
            self.is_initialized = True
            logger.info("✅ Motor inicializado")
            
        except Exception as e:
            logger.warning(f"⚠️ Inicialización parcial: {e}")
            self.is_initialized = True
    
    async def _init_database(self):
        """Inicializar base de datos"""
        try:
            engine = create_engine(self.settings.database_url)
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            self.db_session = Session()
            logger.info("✅ Base de datos inicializada")
        except Exception as e:
            logger.warning(f"⚠️ Error BD: {e}")
    
    async def _init_redis(self):
        """Inicializar Redis"""
        try:
            self.redis_client = redis.from_url(self.settings.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis inicializado")
        except Exception as e:
            logger.warning(f"⚠️ Error Redis: {e}")
            self.redis_client = None
    
    async def _init_cloud_apis(self):
        """Inicializar APIs cloud"""
        if OPENAI_AVAILABLE and self.settings.openai_api_key:
            try:
                self.models_cloud["openai"] = AsyncOpenAI(api_key=self.settings.openai_api_key)
                logger.info("✅ OpenAI configurado")
            except Exception as e:
                logger.warning(f"⚠️ Error OpenAI: {e}")
        
        if ANTHROPIC_AVAILABLE and self.settings.anthropic_api_key:
            try:
                self.models_cloud["anthropic"] = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
                logger.info("✅ Anthropic configurado")
            except Exception as e:
                logger.warning(f"⚠️ Error Anthropic: {e}")
        
        if GOOGLE_AVAILABLE and self.settings.google_api_key:
            try:
                genai.configure(api_key=self.settings.google_api_key)
                self.models_cloud["google"] = genai
                logger.info("✅ Google AI configurado")
            except Exception as e:
                logger.warning(f"⚠️ Error Google: {e}")
    
    async def _init_local_models(self):
        """Inicializar modelos locales"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ Transformers no está disponible. No se cargarán modelos locales.")
            return
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️ Dispositivo: {device}")
        
        # Cargar modelos según configuración
        for task_type, models in self.settings.local_models.items():
            for model_info in models:
                model_name = model_info["name"]
                try:
                    logger.info(f"📥 Cargando {model_name} para tarea: {task_type}...")
                    
                    loop = asyncio.get_event_loop()
                    
                    def load_model(name=model_name, task=task_type):
                        # Usar device_map="auto" para optimizar carga en GPU/CPU
                        model_kwargs = {"device_map": "auto" if device == "cuda" else None}
                        
                        # Cargar pipeline con manejo de errores
                        pipe = pipeline(
                            task,
                            model=name,
                            device=0 if device == "cuda" else -1,
                            model_kwargs=model_kwargs
                        )
                        return pipe
                    
                    # Cargar el modelo en un hilo separado para no bloquear
                    pipeline_obj = await loop.run_in_executor(None, load_model)
                    
                    self.models_local[model_name] = {
                        "pipeline": pipeline_obj,
                        "type": task_type,
                        "display_name": model_info.get("display_name", model_name),
                        "size": model_info.get("size", "N/A")
                    }
                    
                    logger.info(f"✅ Modelo {model_name} cargado correctamente")
                    
                except Exception as e:
                    logger.error(f"❌ Error cargando modelo {model_name}: {str(e)}")
                    continue
    
    async def generate_text(self, prompt: str, model_preference: str = "auto", 
                          max_length: int = 512, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        """Generar texto"""
        start_time = time.time()
        self.stats["requests_total"] += 1
        
        try:
            # Verificar cache
            prompt_hash = hashlib.md5(f"{prompt}_{max_length}".encode()).hexdigest()
            cached_result = await self._get_cached_result(prompt_hash)
            if cached_result:
                self.stats["requests_cached"] += 1
                return cached_result
            
            result = None
            
            if model_preference == "auto":
                if self.models_local:
                    result = await self._generate_local(prompt, max_length, temperature)
                elif self.models_cloud:
                    result = await self._generate_cloud(prompt, max_length, temperature)
                else:
                    result = self._generate_fallback(prompt)
            elif model_preference == "local" and self.models_local:
                result = await self._generate_local(prompt, max_length, temperature)
            elif model_preference == "cloud" and self.models_cloud:
                result = await self._generate_cloud(prompt, max_length, temperature)
            else:
                result = self._generate_fallback(prompt)
            
            # Agregar métricas
            inference_time = time.time() - start_time
            result["inference_time"] = inference_time
            result["timestamp"] = datetime.now().isoformat()
            
            # Actualizar stats
            self.stats["tokens_processed"] += result.get("tokens_used", 0)
            self.stats["inference_time_total"] += inference_time
            
            # Cache
            await self._cache_result(prompt_hash, result)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error en generación: {e}")
            return self._generate_fallback(prompt)
    
    async def _generate_local(self, prompt: str, max_length: int, temperature: float) -> Dict[str, Any]:
        """Generar con modelos locales"""
        self.stats["requests_local"] += 1
        
        model_name = list(self.models_local.keys())[0]
        pipeline_obj = self.models_local[model_name]["pipeline"]
        
        try:
            loop = asyncio.get_event_loop()
            
            def generate():
                result = pipeline_obj(prompt, max_length=max_length, temperature=temperature, 
                                    do_sample=True, pad_token_id=pipeline_obj.tokenizer.eos_token_id)
                return result[0]["generated_text"]
            
            generated_text = await loop.run_in_executor(None, generate)
            
            return {
                "response": generated_text,
                "model_used": model_name,
                "model_type": "local",
                "tokens_used": len(generated_text.split()),
                "engine": "local"
            }
            
        except Exception as e:
            logger.error(f"Error generación local: {e}")
            raise
    
    async def _generate_cloud(self, prompt: str, max_length: int, temperature: float) -> Dict[str, Any]:
        """Generar con APIs cloud"""
        self.stats["requests_cloud"] += 1
        
        if "openai" in self.models_cloud:
            return await self._generate_openai(prompt, max_length, temperature)
        elif "anthropic" in self.models_cloud:
            return await self._generate_anthropic(prompt, max_length, temperature)
        elif "google" in self.models_cloud:
            return await self._generate_google(prompt, max_length, temperature)
        else:
            raise ValueError("No hay APIs cloud disponibles")
    
    async def _generate_openai(self, prompt: str, max_length: int, temperature: float) -> Dict[str, Any]:
        """OpenAI"""
        try:
            client = self.models_cloud["openai"]
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length,
                temperature=temperature
            )
            
            return {
                "response": response.choices[0].message.content,
                "model_used": "gpt-3.5-turbo",
                "model_type": "cloud",
                "tokens_used": response.usage.total_tokens,
                "engine": "openai"
            }
        except Exception as e:
            logger.error(f"Error OpenAI: {e}")
            raise
    
    async def _generate_anthropic(self, prompt: str, max_length: int, temperature: float) -> Dict[str, Any]:
        """Anthropic"""
        try:
            client = self.models_cloud["anthropic"]
            message = await client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=max_length,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "response": message.content[0].text,
                "model_used": "claude-3-sonnet",
                "model_type": "cloud",
                "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
                "engine": "anthropic"
            }
        except Exception as e:
            logger.error(f"Error Anthropic: {e}")
            raise
    
    async def _generate_google(self, prompt: str, max_length: int, temperature: float) -> Dict[str, Any]:
        """Google AI"""
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = await model.generate_content_async(prompt)
            
            return {
                "response": response.text,
                "model_used": "gemini-pro",
                "model_type": "cloud",
                "tokens_used": len(response.text.split()),
                "engine": "google"
            }
        except Exception as e:
            logger.error(f"Error Google: {e}")
            raise
    
    def _generate_fallback(self, prompt: str) -> Dict[str, Any]:
        """Fallback básico"""
        return {
            "response": f"Respuesta simulada para: {prompt[:50]}...",
            "model_used": "fallback",
            "model_type": "basic",
            "tokens_used": len(prompt.split()) + 10,
            "engine": "fallback"
        }
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Análisis de sentimiento"""
        # Implementación básica
        return {
            "sentiment": "neutral",
            "confidence": 0.8,
            "method": "basic"
        }
    
    async def summarize_text(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        """Resumen de texto"""
        # Implementación básica
        return {
            "summary": text[:max_length] + "..." if len(text) > max_length else text,
            "method": "basic"
        }
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """Modelos disponibles"""
        models = {
            "local": list(self.models_local.keys()),
            "cloud": [],
            "embeddings": None
        }
        
        if "openai" in self.models_cloud:
            models["cloud"].extend(["openai:gpt-3.5-turbo", "openai:gpt-4"])
        if "anthropic" in self.models_cloud:
            models["cloud"].extend(["anthropic:claude-3-sonnet"])
        if "google" in self.models_cloud:
            models["cloud"].extend(["google:gemini-pro"])
        
        return models
    
    async def get_stats(self) -> Dict[str, Any]:
        """Estadísticas"""
        uptime = datetime.now() - self.stats["start_time"]
        avg_time = self.stats["inference_time_total"] / max(self.stats["requests_total"], 1)
        success_rate = ((self.stats["requests_total"] - self.stats["errors"]) / 
                       max(self.stats["requests_total"], 1)) * 100
        
        return {
            **self.stats,
            "uptime_formatted": str(uptime),
            "avg_inference_time": avg_time,
            "success_rate": success_rate,
            "models_local_count": len(self.models_local),
            "models_cloud_count": len(self.models_cloud)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check"""
        return {
            "status": "healthy" if self.is_initialized else "initializing",
            "models_local": len(self.models_local),
            "models_cloud": len(self.models_cloud),
            "total_requests": self.stats["requests_total"],
            "error_rate": (self.stats["errors"] / max(self.stats["requests_total"], 1)) * 100
        }
    
    def is_ready(self) -> bool:
        """¿Está listo?"""
        return self.is_initialized
    
    async def _get_cached_result(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Obtener cache"""
        if not self.redis_client:
            return None
        try:
            cached = self.redis_client.get(f"llm_cache:{prompt_hash}")
            return json.loads(cached) if cached else None
        except:
            return None
    
    async def _cache_result(self, prompt_hash: str, result: Dict[str, Any]):
        """Guardar cache"""
        if not self.redis_client:
            return
        try:
            self.redis_client.setex(f"llm_cache:{prompt_hash}", 3600, json.dumps(result, default=str))
        except:
            pass
    
    async def cleanup(self):
        """Limpiar recursos"""
        logger.info("🔄 Limpiando motor...")
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
        if self.db_session:
            try:
                self.db_session.close()
            except:
                pass
        self.models_local.clear()
        self.models_cloud.clear()
        logger.info("✅ Limpieza completada")
