"""Baseline and RL policies for autoscaling."""

from inferscale.policies.base import Policy, PolicyRegistry
from inferscale.policies.fixed import FixedCapacityPolicy
from inferscale.policies.threshold import ThresholdPolicy
from inferscale.policies.queue import QueueBasedPolicy
from inferscale.policies.ppo import PPOPolicy


def get_policy(name: str, config: dict) -> Policy:
    """Get a policy by name."""
    registry = PolicyRegistry()
    registry.register("fixed", FixedCapacityPolicy())
    registry.register("threshold", ThresholdPolicy())
    registry.register("queue", QueueBasedPolicy())
    registry.register("ppo", PPOPolicy())
    return registry.create(name, config)


__all__ = [
    "Policy",
    "PolicyRegistry",
    "FixedCapacityPolicy",
    "ThresholdPolicy",
    "QueueBasedPolicy",
    "PPOPolicy",
    "get_policy",
]
