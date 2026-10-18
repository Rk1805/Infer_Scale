"""Reward function for InferScale RL environment."""

from typing import Dict, Any, Optional

class RewardFunction:
    """Configurable reward function for autoscaling."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.alpha = self.config.get("alpha", 0.1)  # cost weight
        self.beta = self.config.get("beta", 1.0)    # SLA violation weight
        self.gamma = self.config.get("gamma", 0.01) # queue pressure weight
        self.delta = self.config.get("delta", 0.01) # scaling churn weight

    def compute(
        self,
        cost: float,
        sla_violation: bool,
        queue_pressure: float,
        scaling_churn: int,
    ) -> Dict[str, float]:
        """Compute reward components."""
        # Normalize components
        normalized_cost = min(cost, 10.0) / 10.0  # cap at 10x cost
        sla_penalty = 1.0 if sla_violation else 0.0
        normalized_queue = min(queue_pressure, 5.0) / 5.0  # cap at 5x
        normalized_churn = min(scaling_churn, 3.0) / 3.0  # cap at 3 actions

        # Compute reward
        reward = -(
            self.alpha * normalized_cost +
            self.beta * sla_penalty +
            self.gamma * normalized_queue +
            self.delta * normalized_churn
        )

        return {
            "total": reward,
            "cost": -self.alpha * normalized_cost,
            "sla": -self.beta * sla_penalty,
            "queue": -self.gamma * normalized_queue,
            "churn": -self.delta * normalized_churn,
        }

    def reset(self) -> None:
        """Reset reward function state."""
        pass


def compute_normalized_cost(active_replicas: int, max_replicas: int, delta_time: float, cost_per_replica_second: float = 0.01) -> float:
    """Compute normalized compute cost over time interval."""
    return (active_replicas * delta_time * cost_per_replica_second) / (max_replicas * delta_time * cost_per_replica_second)


def compute_sla_violation(end_time: float, arrival_time: float, sla_threshold: float) -> bool:
    """Check if request violated SLA."""
    return (end_time - arrival_time) > sla_threshold


def compute_queue_pressure(queue_length: int, active_replicas: int) -> float:
    """Compute normalized queue pressure."""
    return queue_length / max(1, active_replicas)


def compute_scaling_churn(previous_replicas: int, current_replicas: int) -> int:
    """Compute number of scaling actions taken."""
    return abs(current_replicas - previous_replicas)
