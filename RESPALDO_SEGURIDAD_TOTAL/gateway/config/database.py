#!/usr/bin/env python3
"""
💾 GESTIÓN DE BASE DE DATOS - Agente IA OyP 6.0
===============================================

Sistema de gestión de base de datos para el Gateway.
Configura SQLAlchemy, modelos y sesiones.

Líneas: ~65
"""

import logging
from typing import Generator, Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
import json

from .settings import settings

# ===============================================
# CONFIGURACIÓN BASE
# ===============================================

logger = logging.getLogger("gateway.database")

# Crear engine
engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

# Crear session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base para modelos
Base = declarative_base()

# ===============================================
# MODELOS DE BASE DE DATOS
# ===============================================

class ServiceStatus(Base):
    """Tabla para almacenar estado de servicios"""
    __tablename__ = "service_status"
    
    id = Column(Integer, primary_key=True, index=True)
    service_name = Column(String(50), unique=True, index=True)
    status = Column(String(20))  # healthy, unhealthy, unknown
    last_check = Column(DateTime, default=func.now())
    response_time = Column(Float)  # en milisegundos
    error_message = Column(Text, nullable=True)
    metadata = Column(Text, nullable=True)  # JSON string

class ChatSession(Base):
    """Tabla para sesiones de chat"""
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True)
    user_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=func.now())
    last_activity = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    total_messages = Column(Integer, default=0)

class ChatMessage(Base):
    """Tabla para mensajes de chat"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True)
    message_type = Column(String(20))  # user, assistant, system
    content = Column(Text)
    timestamp = Column(DateTime, default=func.now())
    metadata = Column(Text, nullable=True)  # JSON string

class AnalysisResults(Base):
    """Tabla para resultados de análisis"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(100), unique=True, index=True)
    analysis_type = Column(String(50))  # spss, correlation, regression, etc.
    filename = Column(String(255))
    results = Column(Text)  # JSON string
    created_at = Column(DateTime, default=func.now())
    processing_time = Column(Float)  # en segundos

class FileUploads(Base):
    """Tabla para archivos subidos"""
    __tablename__ = "file_uploads"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String(100), unique=True, index=True)
    original_filename = Column(String(255))
    stored_filename = Column(String(255))
    file_size = Column(Integer)  # en bytes
    file_type = Column(String(50))
    upload_time = Column(DateTime, default=func.now())
    processed = Column(Boolean, default=False)
    analysis_count = Column(Integer, default=0)

# ===============================================
# GESTIÓN DE SESIONES
# ===============================================

def get_database_session() -> Generator[Session, None, None]:
    """
    Generador de sesiones de base de datos.
    Para usar con dependency injection en FastAPI.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Crear todas las tablas en la base de datos"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tablas de base de datos creadas exitosamente")
    except Exception as e:
        logger.error(f"❌ Error creando tablas: {e}")
        raise

def get_db_session() -> Session:
    """Obtener sesión de base de datos (para uso directo)"""
    return SessionLocal()

# ===============================================
# FUNCIONES DE UTILIDAD
# ===============================================

def init_database():
    """Inicializar base de datos completa"""
    logger.info("🔧 Inicializando base de datos...")
    
    try:
        # Crear tablas
        create_tables()
        
        # Verificar conexión
        with get_db_session() as db:
            db.execute("SELECT 1")
            logger.info("✅ Conexión a base de datos verificada")
            
    except Exception as e:
        logger.error(f"❌ Error inicializando base de datos: {e}")
        raise

def reset_database():
    """Resetear base de datos (SOLO DESARROLLO)"""
    if settings.environment != "development":
        raise Exception("Reset solo permitido en desarrollo")
        
    logger.warning("⚠️ Reseteando base de datos...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Base de datos reseteada")

# ===============================================
# INICIALIZACIÓN AUTOMÁTICA
# ===============================================

# Inicializar automáticamente al importar (solo en desarrollo)
if settings.environment == "development":
    try:
        init_database()
    except Exception as e:
        logger.warning(f"⚠️ No se pudo inicializar BD automáticamente: {e}")