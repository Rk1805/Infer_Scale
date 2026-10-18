"""Fixed capacity baseline policy."""

from typing import Any, Dict, Optional

from inferscale.policies.base import Policy


class FixedCapacityPolicy(Policy):
    """Fixed capacity policy - always maintains a constant number of replicas."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.capacity = self.config.get("fixed_capacity", 4)

    def act(self, observation: Dict[str, Any], engine) -> int:
        """Return action to maintain fixed capacity."""
        active = observation.get("active_replicas", 0)
        starting = observation.get("starting_replicas", 0)
        total = active + starting

        # Respect engine limits
        max_replicas = getattr(engine, 'max_replicas', 8)
        min_replicas = getattr(engine, 'min_replicas', 1)
        effective_capacity = min(self.capacity, max_replicas)
        effective_capacity = max(effective_capacity, min_replicas)

        if total < effective_capacity:
            return 2  # scale up
        elif total > effective_capacity:
            return 0  # scale down
        return 1  # hold

    def get_policy_config(self) -> Dict[str, Any]:
        return {"policy": "fixed", "fixed_capacity": self.capacity}

    def create(self, config: Dict[str, Any]) -> "FixedCapacityPolicy":
        return FixedCapacityPolicy(config)
