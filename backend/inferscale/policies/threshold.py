"""Threshold-based autoscaling policy."""

from typing import Any, Dict, Optional
from datetime import datetime

from inferscale.policies.base import Policy


class ThresholdPolicy(Policy):
    """Threshold-based autoscaling using utilization and queue pressure."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.upper_threshold = self.config.get("threshold_upper", 0.75)
        self.lower_threshold = self.config.get("threshold_lower", 0.25)
        self.scaling_in_duration = self.config.get("scaling_in_duration", 60.0)
        self.scale_up_cooldown = self.config.get("cooldown", 30.0)
        self.last_scale_time = -float("inf")
        self.lower_threshold_since: Optional[float] = None

    def act(self, observation: Dict[str, Any], engine) -> int:
        """Select action based on utilization thresholds."""
        current_time = observation.get("time", 0)
        active = observation.get("active_replicas", 0)
        starting = observation.get("starting_replicas", 0)
        total = active + starting
        utilization = observation.get("utilization", 0.0)
        queue_len = observation.get("queue_length", 0)
        queue_pressure = queue_len / max(1, total)

        # Scale up condition: utilization or queue pressure exceeds upper threshold
        if utilization > self.upper_threshold or queue_pressure > 2.0:
            if total < engine.max_replicas and (current_time - self.last_scale_time) > self.scale_up_cooldown:
                self.last_scale_time = current_time
                self.lower_threshold_since = None
                return 2  # scale up

        # Scale down condition: utilization below lower threshold for sufficient duration
        if utilization < self.lower_threshold:
            if self.lower_threshold_since is None:
                self.lower_threshold_since = current_time
            elif current_time - self.lower_threshold_since > self.scaling_in_duration:
                if total > engine.min_replicas and (current_time - self.last_scale_time) > self.scale_up_cooldown:
                    self.last_scale_time = current_time
                    self.lower_threshold_since = None
                    return 0  # scale down
        else:
            self.lower_threshold_since = None

        return 1  # hold

    def get_policy_config(self) -> Dict[str, Any]:
        return {
            "policy": "threshold",
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
            "scaling_in_duration": self.scaling_in_duration,
        }

    def create(self, config: Dict[str, Any]) -> "ThresholdPolicy":
        return ThresholdPolicy(config)
