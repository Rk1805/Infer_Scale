"""HTTP lifecycle tests with real tiny simulations and isolated SQLite/report storage."""

import threading
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from inferscale.api import benchmarks
from inferscale.api.main import app
from inferscale.experiments.persistence import ExperimentDB
from inferscale.services.benchmark import BenchmarkService


@pytest.fixture
def client(tmp_path, monkeypatch):
    database = ExperimentDB(str(tmp_path / "test.db"))
    service = BenchmarkService(database, tmp_path / "reports")
    monkeypatch.setattr(benchmarks, "service", service)
    with TestClient(app) as client:
        yield client


def payload():
    return dict(
        experiment={"episode_duration": 10, "decision_interval": 2, "replica_startup_delay": 1},
        policies=["fixed", "queue"],
        seeds=[11, 22],
        num_requests=10,
    )


def wait_for_completion(client, identifier):
    for _ in range(200):
        response = client.get(f"/api/benchmarks/{identifier}")
        assert response.status_code == 200
        if response.json()["state"] in ("completed", "failed"):
            return response.json()
        time.sleep(0.02)
    pytest.fail("tiny benchmark did not finish within four seconds")


def test_lifecycle_results_and_all_downloads(client):
    response = client.post("/api/benchmarks", json=payload())
    assert response.status_code == 202
    identifier = response.json()["id"]
    job = wait_for_completion(client, identifier)
    assert job["state"] == "completed" and job["progress"] == 1
    data = client.get(f"/api/benchmarks/{identifier}/results").json()
    assert data["successful_runs"] == 4 and data["failed_runs"] == 0
    assert len(data["result"]["runs"]) == 4
    assert data["aggregates"] and data["paired"]
    assert any(j["id"] == identifier for j in client.get("/api/benchmarks").json())
    for artifact in ("runs", "aggregates", "report", "manifest"):
        download = client.get(f"/api/benchmarks/{identifier}/export?artifact={artifact}")
        assert download.status_code == 200
        assert "attachment" in download.headers["content-disposition"]
        assert download.content


@pytest.mark.parametrize(
    "change",
    [
        {"seeds": []},
        {"seeds": [1, 1]},
        {"policies": ["invalid"]},
        {"trace_path": "../../private.csv"},
        {"checkpoint_path": "C:/private.zip"},
        {"experiment": {"decision_interval": 0}},
        {"unknown": True},
        {"experiment": {"dataset_id": "../../private.csv"}},
    ],
)
def test_invalid_requests(client, change):
    assert client.post("/api/benchmarks", json={**payload(), **change}).status_code == 422


def test_missing_checkpoint_and_unknown_identifiers(client):
    assert (
        client.post("/api/benchmarks", json={**payload(), "policies": ["ppo"]}).status_code == 422
    )
    assert client.get(f"/api/benchmarks/{uuid4()}").status_code == 404
    assert client.get("/api/benchmarks/not-a-uuid").status_code == 422
    assert client.get(f"/api/benchmarks/{uuid4()}/export?artifact=../../secret").status_code == 422


def test_pending_results_and_downloads(client):
    identifier = str(uuid4())
    benchmarks.service.db.create_benchmark(identifier, payload())
    assert client.get(f"/api/benchmarks/{identifier}/results").status_code == 409
    assert client.get(f"/api/benchmarks/{identifier}/export").status_code == 409


def test_storage_survives_reopening(client):
    identifier = client.post("/api/benchmarks", json=payload()).json()["id"]
    wait_for_completion(client, identifier)
    reopened = ExperimentDB(benchmarks.service.db.db_path)
    assert reopened.get_benchmark(identifier)["state"] == "completed"
    assert len(reopened.get_benchmark(identifier)["results"]["runs"]) == 4


def test_restart_marks_interrupted_jobs_failed(client):
    identifier = str(uuid4())
    benchmarks.service.db.create_benchmark(identifier, payload())
    benchmarks.service.db.interrupt_benchmarks()
    record = benchmarks.service.db.get_benchmark(identifier)
    assert record["state"] == "failed" and "Server stopped" in record["error"]


def test_all_failed_runs_are_retrievable(client, monkeypatch):
    from inferscale.experiments import benchmark as core

    def fail(*args, **kwargs):
        raise RuntimeError("policy failure")

    monkeypatch.setattr(core, "get_policy", fail)
    identifier = client.post("/api/benchmarks", json=payload()).json()["id"]
    assert wait_for_completion(client, identifier)["state"] == "failed"
    result = client.get(f"/api/benchmarks/{identifier}/results").json()
    assert result["successful_runs"] == 0 and result["failed_runs"] == 4
    assert client.get(f"/api/benchmarks/{identifier}/export").status_code == 200


def test_report_io_failure_has_terminal_status(client, monkeypatch):
    from inferscale.services import benchmark as jobs

    def fail(*args, **kwargs):
        raise OSError("disk fixture failure")

    monkeypatch.setattr(jobs, "write_reports", fail)
    identifier = client.post("/api/benchmarks", json=payload()).json()["id"]
    record = wait_for_completion(client, identifier)
    assert record["state"] == "failed" and "disk fixture failure" in record["error"]
    assert client.get(f"/api/benchmarks/{identifier}/export").status_code == 404


def test_bounded_queue_returns_429(client):
    benchmarks.service._active = 8
    try:
        assert client.post("/api/benchmarks", json=payload()).status_code == 429
    finally:
        benchmarks.service._active = 0


def test_registered_model_outside_model_root_is_rejected(client, tmp_path):
    checkpoint = tmp_path / "outside.zip"
    checkpoint.write_bytes(b"fixture")
    benchmarks.service.db.create_model("outside", "outside", str(checkpoint), {})
    response = client.post(
        "/api/benchmarks", json={**payload(), "policies": ["ppo"], "model_id": "outside"}
    )
    assert response.status_code == 422


def test_server_remains_responsive_during_benchmark(client, monkeypatch):
    from inferscale.services import benchmark as jobs

    entered, release = threading.Event(), threading.Event()
    original = jobs.run_benchmark

    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(jobs, "run_benchmark", paused)
    try:
        response = client.post("/api/benchmarks", json=payload())
        assert response.status_code == 202
        assert entered.wait(2)
        assert client.get("/api/health").status_code == 200
        identifier = response.json()["id"]
        assert client.get(f"/api/benchmarks/{identifier}").json()["state"] == "running"
    finally:
        release.set()
    assert wait_for_completion(client, identifier)["state"] == "completed"
