"""Fast CPU tests for controlled traces and policy execution; no PPO training."""

import json
import random
import subprocess
import sys

import pytest

from inferscale.experiments import benchmark as module
from inferscale.schemas.benchmark import BenchmarkConfig
from inferscale.schemas.experiment import ExperimentConfig


def small_config(**kwargs):
    return BenchmarkConfig(
        experiment=ExperimentConfig(
            episode_duration=20, decision_interval=2, replica_startup_delay=1, scaling_cooldown=2
        ),
        seeds=[11, 22],
        num_requests=20,
        **kwargs,
    )


def test_repeatability_and_shared_trace():
    config = small_config(workloads=["steady", "flash_crowd"])
    first = module.run_benchmark(config)
    second = module.run_benchmark(config)
    assert len(first.runs) == 12
    assert [r.model_dump() for r in first.runs] == [r.model_dump() for r in second.runs]
    for workload in config.workloads:
        for seed in config.seeds:
            group = [r for r in first.runs if (r.workload, r.seed) == (workload, seed)]
            assert len({r.trace_hash for r in group}) == 1
            assert all(r.status == "completed" and r.metrics["arrivals"] > 0 for r in group)
    assert first.runs[0].trace_hash != first.runs[3].trace_hash


def test_generator_does_not_touch_global_rng():
    before = random.getstate()
    requests = module.prepare_workload(small_config(), "mixed", 11)
    assert random.getstate() == before
    assert min(r.arrival_time_seconds for r in requests) == 0


def test_configuration_and_repeated_decisions(monkeypatch):
    seen = []
    original = module.SimulatorEngine

    def capture(**kwargs):
        engine = original(**kwargs)
        seen.append(engine)
        return engine

    monkeypatch.setattr(module, "SimulatorEngine", capture)
    result = module.run_benchmark(small_config(policies=["fixed"]))
    assert all(r.status == "completed" for r in result.runs)
    assert seen[0].startup_delay == 1 and seen[0].cooldown == 2
    assert seen[0].decision_points == list(range(0, 20, 2))
    assert seen[0].seed == 11


def test_actual_simulator_inputs_are_identical_between_policies(monkeypatch):
    traces = []
    original = module.SimulatorEngine

    def capture(**kwargs):
        traces.append([r.model_dump() for r in kwargs["requests"]])
        return original(**kwargs)

    monkeypatch.setattr(module, "SimulatorEngine", capture)
    module.run_benchmark(small_config(workloads=["flash_crowd"]))
    assert traces[0] == traces[1] == traces[2]
    assert traces[3] == traces[4] == traces[5]
    assert traces[0] != traces[3]


def test_missing_checkpoint_is_explicit_failure():
    result = module.run_benchmark(small_config(policies=["fixed", "ppo"]))
    assert [r.status for r in result.runs] == ["completed", "failed"] * 2
    assert all("checkpoint" in r.error for r in result.runs if r.policy == "ppo")


def test_policy_failure_does_not_discard_other_runs(monkeypatch):
    original = module.get_policy

    def policy(name, config):
        if name == "threshold":
            raise RuntimeError("injected failure")
        return original(name, config)

    monkeypatch.setattr(module, "get_policy", policy)
    result = module.run_benchmark(small_config())
    assert sum(r.status == "failed" for r in result.runs) == 2
    assert sum(r.status == "completed" for r in result.runs) == 4


def test_zero_requests_are_not_zero_latency():
    config = small_config(policies=["fixed"]).model_copy(update={"num_requests": 0})
    run = module.run_benchmark(config).runs[0]
    assert run.status == "completed"
    assert run.metrics["arrivals"] == 0
    assert run.metrics["throughput"] == 0
    assert run.metrics["mean_latency_ms"] is None
    assert run.metrics["completed_request_sla_violation_rate"] is None


def test_loaded_checkpoint_uses_existing_policy_cpu(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"fixture")
    calls = []

    class Loaded:
        model = object()

        def act(self, obs, engine):
            return 1

    def load(name, config):
        calls.append(config)
        return Loaded()

    monkeypatch.setattr(module, "get_policy", load)
    result = module.run_benchmark(small_config(policies=["ppo"], checkpoint_path=str(checkpoint)))
    assert all(r.status == "completed" for r in result.runs)
    assert calls[0]["device"] == "cpu"
    assert len(result.provenance["checkpoint_sha256"]) == 64


def test_corrupt_checkpoint_failure(tmp_path):
    checkpoint = tmp_path / "bad.zip"
    checkpoint.write_bytes(b"invalid")
    result = module.run_benchmark(small_config(policies=["ppo"], checkpoint_path=str(checkpoint)))
    assert all(r.status == "failed" and r.error for r in result.runs)


def test_imported_trace_is_normalized(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text("id,arrival_time_seconds,prompt_tokens,output_tokens\na,1700000000,30,10\n")
    config = small_config(policies=["fixed"], trace_path=str(trace))
    result = module.run_benchmark(config)
    assert all(r.metrics["arrivals"] == 1 for r in result.runs)
    assert result.runs[0].trace_hash == result.runs[1].trace_hash


def test_real_checkpoint_inference_without_training(tmp_path):
    """An untrained test fixture verifies actual save/load/predict compatibility only."""
    from stable_baselines3 import PPO

    from inferscale.rl.env import InferScaleEnv

    checkpoint = tmp_path / "fixture.zip"
    env = InferScaleEnv(config={"episode_duration": 10})
    PPO("MlpPolicy", env, n_steps=8, batch_size=4, device="cpu", verbose=0).save(checkpoint)
    result = module.run_benchmark(small_config(policies=["ppo"], checkpoint_path=str(checkpoint)))
    assert all(run.status == "completed" for run in result.runs)
    env.close()


def test_cli_creates_actual_reports(tmp_path):
    output = tmp_path / "result.json"
    reports = tmp_path / "reports"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "inferscale.cli",
            "benchmark",
            "--seeds",
            "11",
            "22",
            "--policies",
            "fixed",
            "queue",
            "--workloads",
            "flash_crowd",
            "--duration",
            "10",
            "--num-requests",
            "12",
            "--output",
            str(output),
            "--reports-dir",
            str(reports),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    data = json.loads(output.read_text())
    assert len(data["runs"]) == 4
    assert all(run["metrics"]["arrivals"] == 12 for run in data["runs"])
    assert len(list(reports.glob("*/report.md"))) == 1


@pytest.mark.parametrize(
    "values",
    [
        {"seeds": []},
        {"seeds": [1, 1]},
        {"policies": ["unknown"]},
        {"experiment": {"decision_interval": 0}},
        {"experiment": {"initial_replicas": 10}},
        {"experiment": {"decision_interval": float("inf")}},
        {"experiment": {"dataset_id": "missing.csv"}},
    ],
)
def test_invalid_benchmark_configuration(values):
    with pytest.raises(ValueError):
        BenchmarkConfig(**values)
