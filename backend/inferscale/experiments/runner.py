"""Experiment orchestration runner."""

import uuid
import threading
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from inferscale.schemas.experiment import ExperimentConfig, ExperimentStatus, PolicyType
from inferscale.simulator.engine import SimulatorEngine, run_simulation
from inferscale.policies import get_policy
from inferscale.data.loader import load_requests


class JobState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExperimentJob:
    """Represents a running experiment job."""
    id: str
    config: ExperimentConfig
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    timeline: List[Dict] = field(default_factory=list)
    cancel_requested: bool = False
    thread: Optional[threading.Thread] = None


class ExperimentRunner:
    """Manages experiment execution."""

    def __init__(self):
        self.jobs: Dict[str, ExperimentJob] = {}
        self._lock = threading.Lock()

    def create_job(self, config: ExperimentConfig) -> ExperimentJob:
        """Create a new experiment job."""
        job = ExperimentJob(
            id=str(uuid.uuid4())[:8],
            config=config,
        )
        with self._lock:
            self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Optional[ExperimentJob]:
        """Get job by ID."""
        with self._lock:
            return self.jobs.get(job_id)

    def list_jobs(self, status: Optional[JobState] = None) -> List[ExperimentJob]:
        """List all jobs, optionally filtered by status."""
        with self._lock:
            jobs = list(self.jobs.values())
            if status:
                jobs = [j for j in jobs if j.state == status]
            return sorted(jobs, key=lambda j: j.started_at or datetime.max, reverse=True)

    def start_job(self, job_id: str) -> bool:
        """Start a job in background thread."""
        job = self.get_job(job_id)
        if not job or job.state != JobState.QUEUED:
            return False

        job.state = JobState.RUNNING
        job.started_at = datetime.now()
        job.thread = threading.Thread(target=self._run_job, args=(job,))
        job.thread.daemon = True
        job.thread.start()
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Request job cancellation."""
        job = self.get_job(job_id)
        if not job or job.state not in [JobState.QUEUED, JobState.RUNNING]:
            return False

        if job.state == JobState.QUEUED:
            job.state = JobState.CANCELLED
            job.cancel_requested = True
            job.completed_at = datetime.now()
            return True

        job.cancel_requested = True
        job.state = JobState.CANCELLING
        return True

    def _run_job(self, job: ExperimentJob) -> None:
        """Execute the experiment job."""
        try:
            config = job.config
            rng = random.Random(config.random_seed)

            # Load requests
            if config.dataset_id == "synthetic":
                from inferscale.data.synth import SYNTHETIC_WORKLOADS
                gen_func = SYNTHETIC_WORKLOADS.get(config.workload_type, SYNTHETIC_WORKLOADS["steady"])
                if config.workload_type == "mixed":
                    # Combine multiple types
                    requests = []
                    for wt in ["steady", "periodic", "flash_crowd"]:
                        requests.extend(SYNTHETIC_WORKLOADS[wt](
                            num_requests=1000, rng=rng, start_time_seconds=0.0))
                else:
                    requests = gen_func(rng=rng, start_time_seconds=0.0)
            else:
                # Load from dataset
                requests = load_requests(config.dataset_id)
                if requests:
                    origin = min(r.arrival_time_seconds for r in requests)
                    requests = [r.model_copy(update={
                        "arrival_time_seconds": r.arrival_time_seconds - origin}) for r in requests]

            # Apply trace window
            if config.trace_window_minutes > 0 and requests:
                start_time = min(r.arrival_time_seconds for r in requests)
                end_time = start_time + config.trace_window_minutes * 60
                requests = [r for r in requests if r.arrival_time_seconds <= end_time]

            # Run simulation with selected policy
            policy = get_policy(config.policy, config.model_dump())

            # Setup simulator
            engine = SimulatorEngine(
                requests=requests,
                initial_replicas=config.initial_replicas,
                min_replicas=config.min_replicas,
                max_replicas=config.max_replicas,
                startup_delay=config.replica_startup_delay,
                cooldown=config.scaling_cooldown,
                decision_interval=config.decision_interval,
                episode_duration=config.episode_duration,
                sla_threshold=config.sla_threshold,
                replica_capacity=config.replica_capacity,
                seed=config.random_seed,
                cost_per_replica_second=config.cost_per_replica_second,
                slo_target=config.slo_target,
            )

            # Run simulation
            events = []
            step_count = 0
            max_steps = int(config.episode_duration / config.decision_interval)

            while engine.events and engine.current_time < config.episode_duration:
                if job.cancel_requested:
                    job.state = JobState.CANCELLED
                    job.completed_at = datetime.now()
                    return

                result = engine.step()
                step_count += 1
                job.progress = min(step_count / max_steps, 1.0)

                if result and result.get("decision_point"):
                    obs = result["observation"]
                    obs["time"] = engine.current_time
                    action = policy.act(obs, engine)
                    engine.take_action(action)
                    events.append({"time": engine.current_time, "action": action, "observation": obs})

            # Get final metrics
            metrics = engine.get_metrics()
            timeline = engine.get_timeline()

            job.results = metrics
            job.timeline = timeline
            job.state = JobState.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0

        except Exception as e:
            job.state = JobState.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()


# Global runner instance
runner = ExperimentRunner()
