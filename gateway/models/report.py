from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Enum
from sqlalchemy.orm import relationship
from . import Base

class Report(Base):
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    report_type = Column(String(50), nullable=False)  # weekly, monthly, final, etc.
    title = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), default='draft')  # draft, generating, completed, failed
    file_path = Column(String(512))  # Path to the generated report file
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    generated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    project = relationship("Project", back_populates="reports")
    
    def __repr__(self):
        return f"<Report(id={self.id}, type='{self.report_type}', status='{self.status}')>"
