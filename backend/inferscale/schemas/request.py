"""Request schema for synthetic and real traffic traces."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, computed_field, field_validator


class RequestSchema(BaseModel):
    """Schema for a single inference request."""

    id: str = Field(..., description="Unique request identifier")
    arrival_time_seconds: float = Field(..., ge=0, description="Timestamp in seconds since epoch")
    prompt_tokens: int = Field(..., ge=0, description="Number of prompt tokens")
    output_tokens: int = Field(..., ge=0, description="Number of output tokens")
    source: Optional[str] = Field(None, description="Source identifier for traceability")

    @field_validator("arrival_time_seconds")
    @classmethod
    def _validate_timestamp(cls, v):
        """Ensure timestamp is valid."""
        if v < 0:
            raise ValueError("arrival_time_seconds must be non-negative")
        return v

    def model_dump_json(self) -> str:
        return super().model_dump_json()


class SyntheticWorkloadType(str):
    """Enumeration of synthetic workload types."""

    STEADY = "steady"
    PERIODIC = "periodic"
    FLASH_CROWD = "flash_crowd"
    MIXED = "mixed"

    @classmethod
    def values(cls) -> List[str]:
        return [cls.STEADY, cls.PERIODIC, cls.FLASH_CROWD, cls.MIXED]
