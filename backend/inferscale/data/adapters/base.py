"""Base adapter interface for data sources."""

from abc import ABC, abstractmethod
from typing import List

class DataAdapter(ABC):
    """Abstract base class for data adapters."""

    @abstractmethod
    def load(self, filepath: str) -> List[dict]:
        """Load raw data from a file."""
        pass

    @abstractmethod
    def validate(self, row: dict) -> bool:
        """Validate a single row record."""
        pass

    @property
    def source_name(self) -> str:
        """Return the name of this adapter's source."""
        return "adaptor"
