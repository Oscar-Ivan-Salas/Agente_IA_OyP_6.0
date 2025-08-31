import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL from environment or default to SQLite
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agente_ia.db")

# For SQLite, we need to handle some special cases
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {},
    poolclass=StaticPool if "sqlite" in SQLALCHEMY_DATABASE_URL else None,
    echo=True  # Set to False in production
)

# Create a configured "Session" class
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# Base class for models
Base = None  # Will be imported from models/__init__.py

def init_db():
    """
    Initialize the database by creating all tables.
    This should be called during application startup.
    """
    from .models import Base
    Base.metadata.create_all(bind=engine)

def get_db():
    """
    Dependency to get DB session.
    Use this in FastAPI route dependencies.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session():
    """
    Get a new DB session.
    Use this when you need a session outside of FastAPI's dependency injection.
    """
    return SessionLocal()
