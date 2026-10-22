"""Validated configuration and portable results for controlled policy evaluation."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inferscale.schemas.experiment import ExperimentConfig

PolicyName = Literal["fixed", "threshold", "queue", "ppo"]
WorkloadName = Literal["steady", "periodic", "flash_crowd", "mixed"]
METRICS = (
    "total_cost",
    "arrivals",
    "completions",
    "sla_violations",
    "max_queue_depth",
    "scaling_events",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_latency_ms",
    "completed_request_sla_violation_rate",
    "throughput",
    "goodput",
    "unfinished_requests",
)


class PolicySettings(BaseModel):
    """Existing baseline policy parameters, shared across benchmark replications."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    fixed_capacity: int = Field(default=4, ge=1, le=100)
    threshold_upper: float = Field(default=0.75, ge=0, le=1)
    threshold_lower: float = Field(default=0.25, ge=0, le=1)
    scaling_in_duration: float = Field(default=60, ge=0)
    queue_based_target: float = Field(default=0.5, gt=0)
    queue_scale_up: float = Field(default=1.0, gt=0)
    queue_scale_down: float = Field(default=0.2, ge=0)

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if self.threshold_lower >= self.threshold_upper:
            raise ValueError("threshold_lower must be below threshold_upper")
        if self.queue_scale_down >= self.queue_scale_up:
            raise ValueError("queue_scale_down must be below queue_scale_up")
        return self


class BenchmarkConfig(BaseModel):
    """A matrix of independent workload seeds and policies on a common simulator config.

    A trace_path is for trusted local Python/CLI callers only. The HTTP schema does
    not expose it. Identical deterministic imported traces are not independent seeds.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    policies: list[PolicyName] = Field(default_factory=lambda: ["fixed", "threshold", "queue"])
    workloads: list[WorkloadName] = Field(default_factory=lambda: ["steady"])
    seeds: list[int] = Field(default_factory=lambda: [11, 22, 33, 44, 55])
    num_requests: int = Field(default=5000, ge=0, le=20000)
    policy_settings: PolicySettings = Field(default_factory=PolicySettings)
    checkpoint_path: str | None = None
    model_id: str | None = None
    trace_path: str | None = None
    evaluation_split: str = Field(default="unverified", max_length=200)

    @model_validator(mode="after")
    def validate_matrix(self):
        for name in ("policies", "workloads", "seeds"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be nonempty and contain no duplicates")
        if any(seed < 0 or seed >= 2**32 for seed in self.seeds):
            raise ValueError("seeds must be integers between 0 and 2**32 - 1")
        if len(self.policies) * len(self.workloads) * len(self.seeds) > 500:
            raise ValueError("A benchmark supports at most 500 runs")
        cfg = self.experiment
        if cfg.dataset_id != "synthetic" and not self.trace_path:
            raise ValueError("Use trace_path for a canonical imported dataset")
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in cfg.model_dump().values()
        ):
            raise ValueError("Experiment settings must contain finite numbers")
        if cfg.decision_interval <= 0 or cfg.episode_duration / cfg.decision_interval > 100000:
            raise ValueError("Use a positive decision interval and at most 100000 decisions")
        if cfg.episode_duration > 86400 or cfg.max_replicas > 100:
            raise ValueError("Maximum duration is 86400 seconds and maximum replicas is 100")
        if not cfg.min_replicas <= cfg.initial_replicas <= cfg.max_replicas:
            raise ValueError("initial_replicas must be between min_replicas and max_replicas")
        if self.trace_path and len(self.workloads) != 1:
            raise ValueError("An imported trace must have a single workload label")
        return self


class BenchmarkRun(BaseModel):
    """One policy on one preserved workload; failures remain part of the record."""

    policy: PolicyName
    workload: WorkloadName
    seed: int
    status: Literal["completed", "failed"]
    trace_hash: str | None = None
    request_count: int = 0
    config: dict = Field(default_factory=dict)
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    error: str | None = None


class BenchmarkResult(BaseModel):
    """Portable raw benchmark result, independent of persistence or the web server."""

    config: BenchmarkConfig
    runs: list[BenchmarkRun]
    provenance: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
