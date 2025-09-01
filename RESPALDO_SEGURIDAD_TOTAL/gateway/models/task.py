from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.orm import relationship
from . import Base

class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    parent_id = Column(Integer, ForeignKey('tasks.id', ondelete='CASCADE'), nullable=True)
    wbs = Column(String(50), nullable=False)  # Work Breakdown Structure code (e.g., 1.2.3)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), default='pending')  # pending, in_progress, completed, blocked
    priority = Column(String(20), default='medium')  # low, medium, high, critical
    progress = Column(Float, default=0.0)  # 0-100%
    estimated_hours = Column(Float, default=0.0)
    actual_hours = Column(Float, default=0.0)
    start_date = Column(DateTime, nullable=True)
    due_date = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    project = relationship("Project", back_populates="tasks")
    parent = relationship("Task", remote_side=[id], back_populates="subtasks")
    subtasks = relationship("Task", back_populates="parent", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Task(id={self.id}, name='{self.name}', status='{self.status}')>"
