from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from . import Base

class DailyLog(Base):
    __tablename__ = 'daily_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    log_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    summary = Column(Text, nullable=False)
    details = Column(Text)
    hours_worked = Column(Float, default=0.0)
    blockers = Column(Text)
    next_steps = Column(Text)
    mood = Column(String(50))  # Optional: good, neutral, bad
    additional_data = Column(JSON, nullable=True)  # For any extra structured data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    project = relationship("Project", back_populates="daily_logs")
    
    def __repr__(self):
        return f"<DailyLog(id={self.id}, project_id={self.project_id}, date='{self.log_date}')>"
