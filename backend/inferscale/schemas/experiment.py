"""Experiment configuration and metrics schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, computed_field
from datetime import datetime

from inferscale.config import settings

class PolicyType(str):
    """Enumeration of supported policies."""

    FIXED = "fixed"
    THRESHOLD = "threshold"
    QUEUE = "queue"
    PPO = "ppo"

    @classmethod
    def values(cls) -> List[str]:
        return [cls.FIXED, cls.THRESHOLD, cls.QUEUE, cls.PPO]


class ExperimentStatus(str):
    """Enumeration of experiment statuses."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @classmethod
    def values(cls) -> List[str]:
        return [cls.QUEUED, cls.RUNNING, cls.COMPLETED, cls.CANCELLED, cls.FAILED]


class ExperimentConfig(BaseModel):
    """Configuration for an experiment run."""

    workload_type: str = Field(default="steady", description="Workload type")
    dataset_id: str = Field(default="synthetic", description="Dataset identifier")
    trace_window_minutes: float = Field(default=30.0, ge=1.0, description="Trace window in minutes")
    policy: str = Field(default="threshold", description="Policy type")
    initial_replicas: int = Field(default=2, ge=1, description="Initial number of replicas")
    min_replicas: int = Field(default=1, ge=1, description="Minimum replicas")
    max_replicas: int = Field(default=8, ge=1, description="Maximum replicas")
    replica_startup_delay: float = Field(default=10.0, ge=0.0, description="Startup delay in seconds")
    scaling_cooldown: float = Field(default=30.0, ge=0.0, description="Scaling cooldown in seconds")
    decision_interval: float = Field(
        default=5.0, gt=0.0, description="Decision interval in seconds"
    )
    episode_duration: float = Field(default=3600.0, ge=1.0, description="Episode duration in seconds")
    sla_threshold: float = Field(default=30.0, ge=1.0, description="SLA threshold in seconds")
    slo_target: float = Field(
        default=0.99, ge=0.0, le=1.0, allow_inf_nan=False,
        description="Required fraction of arrivals completed within the SLA",
    )
    cost_per_replica_second: float = Field(
        default=settings.gpu_cost_per_replica_second, ge=0.0, allow_inf_nan=False,
        description="USD per second in starting, active or draining state",
    )
    random_seed: int = Field(default=42, description="Random seed")
    replica_capacity: int = Field(default=10, ge=1, description="Replica capacity (concurrent requests)")
    reward_alpha: float = Field(default=0.1, ge=0.0, description="Cost weight")
    reward_beta: float = Field(default=1.0, ge=0.0, description="SLA violation weight")
    reward_gamma: float = Field(default=0.01, ge=0.0, description="Queue pressure weight")
    reward_delta: float = Field(default=0.01, ge=0.0, description="Scaling churn weight")
    ppo_timesteps: int = Field(default=100000, ge=1000, description="PPO training timesteps")
    ppo_alpha: float = Field(default=0.1, description="PPO reward alpha")
    ppo_beta: float = Field(default=0.01, description="PPO reward beta")
    ppo_gamma: float = Field(default=0.99, description="PPO gamma")
    short_run: bool = Field(default=True, description="Short run mode")

    @field_validator("workload_type")
    @classmethod
    def _validate_workload_type(cls, v):
        """Validate workload type."""
        valid_types = ["steady", "periodic", "flash_crowd", "mixed"]
        if v not in valid_types:
            raise ValueError(f"Invalid workload_type: {v}. Must be one of {valid_types}")
        return v

    @field_validator("policy")
    @classmethod
    def _validate_policy(cls, v):
        """Validate policy type."""
        if v not in PolicyType.values():
            raise ValueError(f"Invalid policy: {v}")
        return v

    @field_validator("max_replicas")
    @classmethod
    def _validate_max_replicas(cls, v, info):
        """Validate max replicas >= min replicas."""
        if hasattr(info, 'data') and info.data.get("min_replicas") is not None:
            if v < info.data["min_replicas"]:
                raise ValueError("max_replicas must be >= min_replicas")
        return v

    class Config:
        extra = "forbid"


class ExperimentResult(BaseModel):
    """Results from an experiment run."""

    experiment_id: str = Field(..., description="Experiment ID")
    total_cost: float = Field(..., ge=0, description="Total compute cost")
    avg_active_replicas: float = Field(..., ge=0, description="Average active replicas")
    p50_latency_ms: float = Field(..., ge=0, description="p50 latency in ms")
    p95_latency_ms: float = Field(..., ge=0, description="p95 latency in ms")
    p99_latency_ms: float = Field(..., ge=0, description="p99 latency in ms")
    sla_violation_rate: Optional[float] = Field(
        ..., ge=0, le=1,
        description="Violations among resolved requests; unavailable without evidence",
    )
    goodput: float = Field(..., ge=0, description="Goodput")
    throughput: float = Field(..., ge=0, description="Throughput")
    max_queue_depth: float = Field(..., ge=0, description="Max queue depth")
    avg_queue_depth: float = Field(..., ge=0, description="Average queue depth")
    scaling_events: int = Field(..., ge=0, description="Number of scaling events")
    burst_recovery_time: float = Field(default=0.0, ge=0, description="Burst recovery time in seconds")
    unfinished_requests: int = Field(default=0, ge=0, description="Number of unfinished requests")
    episode_reward: float = Field(default=0.0, description="Episode reward")
    timestamp: datetime = Field(default_factory=datetime.now, description="Completion timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
