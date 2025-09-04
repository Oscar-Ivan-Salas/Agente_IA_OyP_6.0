from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Import all models here to ensure they are registered with the Base
from .project import Project
from .task import Task
from .daily_log import DailyLog
from .risk import Risk
from .report import Report

__all__ = ['Base', 'Project', 'Task', 'DailyLog', 'Risk', 'Report']
