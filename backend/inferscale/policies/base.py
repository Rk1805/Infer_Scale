"""Base policy interface and common utilities."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Policy(ABC):
    """Base class for all scaling policies."""

    @abstractmethod
    def act(self, observation: Dict[str, Any], engine) -> int:
        """
        Select an action based on the observation.
        Returns an integer action code.
        """
        pass

    @abstractmethod
    def get_policy_config(self) -> Dict[str, Any]:
        """Get the policy configuration."""
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate policy configuration."""
        return True


class PolicyRegistry:
    """Registry of available policies."""

    def __init__(self):
        self._registry: Dict[str, Policy] = {}

    def register(self, name: str, policy: Policy) -> None:
        self._registry[name] = policy

    def create(self, name: str, config: Dict[str, Any]) -> Policy:
        """Create a policy instance from config."""
        if name not in self._registry:
            raise ValueError(f"Unknown policy: {name}")
        return self._registry[name].create(config)

    def list_policies(self) -> List[str]:
        return list(self._registry.keys())
