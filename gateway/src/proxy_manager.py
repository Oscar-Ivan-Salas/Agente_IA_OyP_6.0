#!/usr/bin/env python3
"""
🔗 GESTOR DE PROXY - Agente IA OyP 6.0
=====================================

Sistema de comunicación con microservicios.
Gestiona proxy, health checks, fallbacks y circuit breakers.

Líneas: ~125
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import httpx
from fastapi import HTTPException
import json

from ..config.settings import settings

# ===============================================
# CONFIGURACIÓN
# ===============================================

logger = logging.getLogger("gateway.proxy_manager")

class CircuitBreakerState:
    """Estados del circuit breaker"""
    CLOSED = "closed"      # Normal - permite requests
    OPEN = "open"          # Fallido - bloquea requests
    HALF_OPEN = "half_open"  # Probando - permite requests limitados

class ServiceHealthStatus:
    """Estados de salud de servicios"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"

# ===============================================
# PROXY MANAGER PRINCIPAL
# ===============================================

class ProxyManager:
    """Gestor principal de proxy a microservicios"""
    
    def __init__(self):
        self.services = settings.services_config
        self.client = httpx.AsyncClient(timeout=settings.http_timeout)
        
        # Circuit breakers por servicio
        self.circuit_breakers: Dict[str, Dict] = {}
        
        # Cache de estado de servicios
        self.service_health: Dict[str, Dict] = {}
        
        # Métricas
        self.metrics: Dict[str, Dict] = {}
        
        # Inicializar
        self._init_circuit_breakers()
        self._init_health_cache()
        
        logger.info(f"🔗 ProxyManager inicializado para {len(self.services)} servicios")
    
    def _init_circuit_breakers(self):
        """Inicializar circuit breakers para cada servicio"""
        for service_id in self.services.keys():
            self.circuit_breakers[service_id] = {
                "state": CircuitBreakerState.CLOSED,
                "failure_count": 0,
                "last_failure": None,
                "next_attempt": None,
                "failure_threshold": 5,  # fallos antes de abrir
                "timeout_duration": 60,  # segundos cerrado
                "success_threshold": 3   # éxitos para cerrar
            }
    
    def _init_health_cache(self):
        """Inicializar cache de salud de servicios"""
        for service_id, config in self.services.items():
            self.service_health[service_id] = {
                "status": ServiceHealthStatus.UNKNOWN,
                "last_check": None,
                "response_time": None,
                "error_message": None,
                "consecutive_failures": 0
            }
            
            self.metrics[service_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_response_time": 0.0,
                "last_request": None
            }
    
    async def proxy_request(
        self,
        service_id: str,
        method: str,
        path: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        json_data: Optional[Any] = None,
        files: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Proxear request a microservicio con circuit breaker
        """
        
        # Verificar si servicio existe
        if service_id not in self.services:
            raise HTTPException(
                status_code=404, 
                detail=f"Servicio '{service_id}' no encontrado"
            )
        
        # Verificar circuit breaker
        if not self._can_make_request(service_id):
            raise HTTPException(
                status_code=503,
                detail=f"Servicio '{service_id}' temporalmente no disponible (Circuit Breaker OPEN)"
            )
        
        service_config = self.services[service_id]
        url = f"{service_config['url']}{path}"
        
        start_time = datetime.now()
        
        try:
            # Preparar headers
            request_headers = headers or {}
            request_headers["X-Gateway-Service"] = service_id
            request_headers["X-Gateway-Timestamp"] = start_time.isoformat()
            
            # Realizar request
            response = await self.client.request(
                method=method.upper(),
                url=url,
                headers=request_headers,
                params=params,
                json=json_data,
                files=files
            )
            
            # Calcular tiempo de respuesta
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Actualizar métricas de éxito
            self._record_success(service_id, response_time)
            
            # Preparar respuesta
            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "service": service_id,
                "response_time_ms": response_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Intentar parsear JSON
            try:
                result["data"] = response.json()
            except:
                result["data"] = response.text
            
            logger.debug(f"✅ Proxy {method} {service_id}{path} -> {response.status_code} ({response_time:.1f}ms)")
            
            return result
            
        except httpx.TimeoutException:
            error_msg = f"Timeout en servicio {service_id}"
            self._record_failure(service_id, error_msg)
            logger.error(f"⏰ {error_msg}")
            raise HTTPException(status_code=504, detail=error_msg)
            
        except httpx.ConnectError:
            error_msg = f"Error de conexión a servicio {service_id}"
            self._record_failure(service_id, error_msg)
            logger.error(f"🔌 {error_msg}")
            raise HTTPException(status_code=503, detail=error_msg)
            
        except Exception as e:
            error_msg = f"Error en proxy a {service_id}: {str(e)}"
            self._record_failure(service_id, error_msg)
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
    
    async def health_check(self, service_id: str) -> Dict[str, Any]:
        """Verificar salud de un servicio específico"""
        
        if service_id not in self.services:
            return {
                "service": service_id,
                "status": ServiceHealthStatus.UNKNOWN,
                "error": "Servicio no configurado"
            }
        
        service_config = self.services[service_id]
        health_url = f"{service_config['url']}{service_config['health_endpoint']}"
        
        start_time = datetime.now()
        
        try:
            response = await self.client.get(health_url, timeout=10.0)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                status = ServiceHealthStatus.HEALTHY
                self.service_health[service_id]["consecutive_failures"] = 0
            else:
                status = ServiceHealthStatus.DEGRADED
                
            # Actualizar cache
            self.service_health[service_id].update({
                "status": status,
                "last_check": datetime.now(),
                "response_time": response_time,
                "error_message": None
            })
            
            try:
                health_data = response.json()
            except:
                health_data = {"status": "ok"}
            
            return {
                "service": service_id,
                "status": status,
                "response_time_ms": response_time,
                "data": health_data,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Actualizar cache con error
            self.service_health[service_id].update({
                "status": ServiceHealthStatus.UNHEALTHY,
                "last_check": datetime.now(),
                "response_time": None,
                "error_message": str(e),
                "consecutive_failures": self.service_health[service_id]["consecutive_failures"] + 1
            })
            
            return {
                "service": service_id,
                "status": ServiceHealthStatus.UNHEALTHY,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Verificar salud de todos los servicios"""
        
        logger.info("🏥 Realizando health check de todos los servicios...")
        
        # Ejecutar health checks en paralelo
        tasks = [
            self.health_check(service_id) 
            for service_id in self.services.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        service_results = {}
        healthy_count = 0
        
        for i, result in enumerate(results):
            service_id = list(self.services.keys())[i]
            
            if isinstance(result, Exception):
                service_results[service_id] = {
                    "service": service_id,
                    "status": ServiceHealthStatus.UNHEALTHY,
                    "error": str(result)
                }
            else:
                service_results[service_id] = result
                if result.get("status") == ServiceHealthStatus.HEALTHY:
                    healthy_count += 1
        
        # Calcular estado general
        total_services = len(self.services)
        overall_status = "healthy" if healthy_count == total_services else \
                        "degraded" if healthy_count > 0 else "unhealthy"
        
        return {
            "overall_status": overall_status,
            "healthy_services": healthy_count,
            "total_services": total_services,
            "services": service_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _can_make_request(self, service_id: str) -> bool:
        """Verificar si circuit breaker permite request"""
        cb = self.circuit_breakers[service_id]
        
        if cb["state"] == CircuitBreakerState.CLOSED:
            return True
        elif cb["state"] == CircuitBreakerState.OPEN:
            # Verificar si es tiempo de intentar
            if cb["next_attempt"] and datetime.now() >= cb["next_attempt"]:
                cb["state"] = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def _record_success(self, service_id: str, response_time: float):
        """Registrar request exitoso"""
        # Actualizar métricas
        metrics = self.metrics[service_id]
        metrics["total_requests"] += 1
        metrics["successful_requests"] += 1
        metrics["last_request"] = datetime.now()
        
        # Actualizar promedio de tiempo de respuesta
        total_successful = metrics["successful_requests"]
        current_avg = metrics["avg_response_time"]
        metrics["avg_response_time"] = ((current_avg * (total_successful - 1)) + response_time) / total_successful
        
        # Resetear circuit breaker si estaba en HALF_OPEN
        cb = self.circuit_breakers[service_id]
        if cb["state"] == CircuitBreakerState.HALF_OPEN:
            cb["state"] = CircuitBreakerState.CLOSED
            cb["failure_count"] = 0
    
    def _record_failure(self, service_id: str, error_message: str):
        """Registrar request fallido"""
        # Actualizar métricas
        metrics = self.metrics[service_id]
        metrics["total_requests"] += 1
        metrics["failed_requests"] += 1
        metrics["last_request"] = datetime.now()
        
        # Actualizar circuit breaker
        cb = self.circuit_breakers[service_id]
        cb["failure_count"] += 1
        cb["last_failure"] = datetime.now()
        
        # Abrir circuit breaker si se alcanza threshold
        if cb["failure_count"] >= cb["failure_threshold"]:
            cb["state"] = CircuitBreakerState.OPEN
            cb["next_attempt"] = datetime.now() + timedelta(seconds=cb["timeout_duration"])
            logger.warning(f"🚨 Circuit breaker ABIERTO para {service_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de todos los servicios"""
        return {
            "services_health": self.service_health,
            "services_metrics": self.metrics,
            "circuit_breakers": {
                service_id: {
                    "state": cb["state"],
                    "failure_count": cb["failure_count"]
                }
                for service_id, cb in self.circuit_breakers.items()
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def close(self):
        """Cerrar cliente HTTP"""
        await self.client.aclose()
        logger.info("✅ ProxyManager cerrado")

# ===============================================
# INSTANCIA GLOBAL
# ===============================================

proxy_manager = ProxyManager()
