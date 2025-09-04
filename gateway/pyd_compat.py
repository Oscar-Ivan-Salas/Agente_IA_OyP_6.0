# gateway/pyd_compat.py
from pydantic.v1 import (
    BaseModel, BaseSettings, Field,
    validator, root_validator, ValidationError
)
