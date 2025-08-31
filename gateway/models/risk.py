from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Enum, Float
from sqlalchemy.orm import relationship
from . import Base

class Risk(Base):
    __tablename__ = 'risks'
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))  # e.g., technical, schedule, resource, external
    probability = Column(Float)  # 0.0 to 1.0
    impact = Column(Float)  # 0.0 to 1.0
    severity = Column(String(20))  # low, medium, high, critical
    status = Column(String(50), default='identified')  # identified, monitored, mitigated, occurred, closed
    mitigation_plan = Column(Text)
    contingency_plan = Column(Text)
    due_date = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    project = relationship("Project", back_populates="risks")
    
    def __repr__(self):
        return f"<Risk(id={self.id}, title='{self.title}', severity='{self.severity}')>"
    
    @property
    def risk_score(self):
        """Calculate the risk score (probability * impact)."""
        if self.probability is not None and self.impact is not None:
            return self.probability * self.impact
        return None
