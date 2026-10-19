"""Reproducible evaluation using the existing simulator and policy implementations."""

import hashlib
import json
import math
import random
import subprocess
from collections.abc import Callable
from pathlib import Path

from inferscale.data.loader import load_requests
from inferscale.data.synth import SYNTHETIC_WORKLOADS
from inferscale.policies import get_policy
from inferscale.schemas.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRun
from inferscale.schemas.request import RequestSchema
from inferscale.simulator.engine import SimulatorEngine

METRIC_NOTES = [
    "Simulation output uses an analytical service model; these are not measured GPU results.",
    "total_cost is the legacy simulator cost: replica startup charges only, not full runtime cost.",
    "Latency and completed_request_sla_violation_rate describe completed requests only; "
    "unfinished_requests must also be considered (horizon censoring).",
    "Legacy avg_active_replicas and utilization are not time-averaged occupancy and are excluded.",
    "Workload variation currently randomizes token counts; arrival patterns are deterministic.",
]


def prepare_workload(config: BenchmarkConfig, workload: str, seed: int) -> list[RequestSchema]:
    """Generate once using a local RNG, or normalize a canonical CSV/JSON trace."""
    if config.trace_path:
        requests = load_requests(config.trace_path)
        origin = min((r.arrival_time_seconds for r in requests), default=0)
        requests = [
            r.model_copy(update={"arrival_time_seconds": r.arrival_time_seconds - origin})
            for r in requests
        ]
    else:
        rng = random.Random(seed)
        kinds = ["steady", "periodic", "flash_crowd"] if workload == "mixed" else [workload]
        requests = []
        for index, kind in enumerate(kinds):
            count = config.num_requests // len(kinds) + (index < config.num_requests % len(kinds))
            requests.extend(
                SYNTHETIC_WORKLOADS[kind](num_requests=count, rng=rng, start_time_seconds=0.0)
            )
    if any(not math.isfinite(r.arrival_time_seconds) for r in requests):
        raise ValueError("Trace timestamps must be finite")
    horizon = min(config.experiment.episode_duration, config.experiment.trace_window_minutes * 60)
    requests = sorted(
        (r for r in requests if r.arrival_time_seconds < horizon),
        key=lambda r: (r.arrival_time_seconds, r.id),
    )
    if len({r.id for r in requests}) != len(requests):
        raise ValueError("Trace request IDs must be unique")
    if any(not math.isfinite(r.arrival_time_seconds) for r in requests):
        raise ValueError("Trace timestamps must be finite")
    return requests


def trace_digest(requests: list[RequestSchema]) -> str:
    """Hash the complete canonical trace shared by the compared policies."""
    payload = json.dumps([r.model_dump() for r in requests], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_metrics(engine: SimulatorEngine) -> dict[str, float | int | None]:
    """Expose measured scalars and explicitly defined completion-based derivatives.

    Existing percentile definitions and legacy total_cost remain unchanged. Missing
    latency and SLA observations are unavailable, rather than fabricated zeroes.
    """
    raw = engine.get_metrics()
    completed = engine.metrics["completions"]
    latencies = engine.metrics["latencies"]
    metrics = {
        name: raw.get(name)
        for name in (
            "total_cost",
            "arrivals",
            "completions",
            "sla_violations",
            "max_queue_depth",
            "scaling_events",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
        )
    }
    metrics.update(
        mean_latency_ms=sum(latencies) / len(latencies) * 1000 if latencies else None,
        completed_request_sla_violation_rate=(
            engine.metrics["sla_violations"] / completed if completed else None
        ),
        throughput=completed / engine.episode_duration,
        goodput=(completed - engine.metrics["sla_violations"]) / engine.episode_duration,
        unfinished_requests=engine.metrics["arrivals"] - completed,
    )
    return metrics


def code_provenance() -> dict:
    """Best-effort Git provenance; never change repository configuration or history."""
    root = Path(__file__).resolve().parents[3]
    try:

        def git(*args):
            return subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()

        return {
            "revision": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "working_tree_dirty": None}


def run_benchmark(
    config: BenchmarkConfig,
    on_run: Callable[[BenchmarkRun, int, int], None] | None = None,
) -> BenchmarkResult:
    """Evaluate every workload/seed/policy without training or silently substituting PPO.

    A failed workload or individual policy run is recorded and does not discard
    other runs. The optional callback receives completed run count for job progress.
    """
    runs: list[BenchmarkRun] = []
    provenance = code_provenance()
    provenance["evaluation_split"] = config.evaluation_split
    warnings = list(METRIC_NOTES)
    if config.evaluation_split == "unverified":
        warnings.append("Training/evaluation separation is unverified; no held-out claim is made.")
    if config.trace_path:
        warnings.append(
            "Imported trace repeats are deterministic; seed repeats are not "
            "independent experimental replications and receive no confidence interval."
        )
    checkpoint_error = None
    if "ppo" in config.policies:
        try:
            checkpoint = Path(config.checkpoint_path or "")
            if not config.checkpoint_path or not checkpoint.is_file():
                raise ValueError(
                    "PPO requires an existing trained .zip checkpoint; no fallback is used"
                )
            provenance["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            provenance["checkpoint_name"] = checkpoint.name
        except (OSError, ValueError) as exc:
            checkpoint_error = str(exc)
    total = len(config.workloads) * len(config.seeds) * len(config.policies)
    for workload in config.workloads:
        for seed in config.seeds:
            requests, workload_error = [], None
            try:
                requests = prepare_workload(config, workload, seed)
                digest = trace_digest(requests)
            except Exception as exc:
                digest, workload_error = None, f"Workload preparation failed: {exc}"
            for policy_name in config.policies:
                effective = config.experiment.model_dump()
                effective.update(config.policy_settings.model_dump())
                effective.update(
                    policy=policy_name,
                    workload_type=workload,
                    random_seed=seed,
                    startup_delay=config.experiment.replica_startup_delay,
                    cooldown=config.experiment.scaling_cooldown,
                )
                run = BenchmarkRun(
                    policy=policy_name,
                    workload=workload,
                    seed=seed,
                    status="failed",
                    config=effective,
                    trace_hash=digest,
                    request_count=len(requests),
                )
                try:
                    if workload_error:
                        raise ValueError(workload_error)
                    if policy_name == "ppo" and checkpoint_error:
                        raise ValueError(checkpoint_error)
                    policy_config = {
                        **effective,
                        "model_path": config.checkpoint_path,
                        "device": "cpu",
                    }
                    policy = get_policy(policy_name, policy_config)
                    if policy_name == "ppo" and getattr(policy, "model", None) is None:
                        raise ValueError(
                            "PPO checkpoint did not load; fallback evaluation is forbidden"
                        )
                    engine = SimulatorEngine(
                        requests=[r.model_copy(deep=True) for r in requests],
                        **{
                            key: effective[key]
                            for key in (
                                "initial_replicas",
                                "min_replicas",
                                "max_replicas",
                                "startup_delay",
                                "cooldown",
                                "decision_interval",
                                "episode_duration",
                                "sla_threshold",
                                "replica_capacity",
                            )
                        },
                        seed=seed,
                    )
                    while engine.events and engine.current_time < engine.episode_duration:
                        event = engine.step()
                        if event and event.get("decision_point"):
                            obs = {**event["observation"], "time": engine.current_time}
                            engine.take_action(policy.act(obs, engine))
                    run.metrics = extract_metrics(engine)
                    run.status = "completed"
                except Exception as exc:
                    run.error = f"{type(exc).__name__}: {exc}"
                runs.append(run)
                if on_run:
                    on_run(run, len(runs), total)
    return BenchmarkResult(config=config, runs=runs, provenance=provenance, warnings=warnings)
