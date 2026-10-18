"""Queue-based reactive autoscaling policy."""

from typing import Any, Dict, Optional

from inferscale.policies.base import Policy


class QueueBasedPolicy(Policy):
    """Queue-based autoscaling based on estimated waiting work per replica."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.queue_target = self.config.get("queue_based_target", 0.5)  # work units per replica
        self.scale_up_threshold = self.config.get("queue_scale_up", 1.0)
        self.scale_down_threshold = self.config.get("queue_scale_down", 0.2)
        self.cooldown = self.config.get("cooldown", 30.0)
        self.last_scale_time = -float("inf")

    def act(self, observation: Dict[str, Any], engine) -> int:
        """Select action based on queue pressure."""
        current_time = observation.get("time", 0)
        active = observation.get("active_replicas", 0)
        starting = observation.get("starting_replicas", 0)
        total = active + starting
        queue_len = observation.get("queue_length", 0)
        estimated_work = observation.get("estimated_work", 0)

        # Work per replica (estimated tokens waiting per active replica)
        work_per_replica = estimated_work / max(1, total)

        # Scale up if work per replica exceeds threshold
        if work_per_replica > self.scale_up_threshold:
            if total < engine.max_replicas and (current_time - self.last_scale_time) > self.cooldown:
                self.last_scale_time = current_time
                return 2  # scale up

        # Scale down if work per replica well below threshold
        if work_per_replica < self.scale_down_threshold:
            if total > engine.min_replicas and (current_time - self.last_scale_time) > self.cooldown * 2:
                self.last_scale_time = current_time
                return 0  # scale down

        return 1  # hold

    def get_policy_config(self) -> Dict[str, Any]:
        return {
            "policy": "queue",
            "queue_target": self.queue_target,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
        }

    def create(self, config: Dict[str, Any]) -> "QueueBasedPolicy":
        return QueueBasedPolicy(config)
