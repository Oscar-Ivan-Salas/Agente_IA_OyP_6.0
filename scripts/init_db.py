#!/usr/bin/env python3
"""
Database initialization script.
This script creates the initial database and runs any pending migrations.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def init_database():
    """Initialize the database and run migrations."""
    from gateway.database import init_db, engine
    from gateway.models import Base
    
    print("Initializing database...")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    print("Database initialized successfully!")
    print("You can now run migrations with: alembic revision --autogenerate -m 'description'")
    print("Then apply them with: alembic upgrade head")

if __name__ == "__main__":
    init_database()
