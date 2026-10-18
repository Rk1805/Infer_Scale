"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from inferscale.api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


class TestDatasetEndpoints:
    """Tests for dataset endpoints."""

    def test_list_datasets(self):
        response = client.get("/api/datasets")
        assert response.status_code == 200
        datasets = response.json()
        assert isinstance(datasets, list)
        assert len(datasets) >= 1  # At least synthetic
        assert any(d["id"] == "synthetic" for d in datasets)

    def test_get_synthetic_dataset_stats(self):
        response = client.get("/api/datasets/synthetic/stats")
        assert response.status_code == 200
        stats = response.json()
        assert stats["id"] == "synthetic"
        assert "record_count" in stats
        assert "prompt_tokens" in stats
        assert "arrival_rate" in stats


class TestExperimentEndpoints:
    """Tests for experiment endpoints."""

    def test_create_experiment(self):
        config = {
            "workload_type": "steady",
            "dataset_id": "synthetic",
            "trace_window_minutes": 5,
            "policy": "threshold",
            "initial_replicas": 2,
            "min_replicas": 1,
            "max_replicas": 4,
            "replica_startup_delay": 5,
            "scaling_cooldown": 10,
            "decision_interval": 2,
            "episode_duration": 30,
            "sla_threshold": 30,
            "random_seed": 42,
            "reward_alpha": 0.1,
            "reward_beta": 1.0,
            "reward_gamma": 0.01,
            "reward_delta": 0.01,
            "ppo_timesteps": 1000,
            "ppo_alpha": 0.1,
            "ppo_beta": 0.01,
            "ppo_gamma": 0.99,
            "short_run": True,
        }
        
        response = client.post("/api/experiments", json=config)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        # State could be queued, running, or completed (short runs finish fast)
        assert data["state"] in ["queued", "running", "completed"]
        assert data["config"]["policy"] == "threshold"

    def test_list_experiments(self):
        response = client.get("/api/experiments")
        assert response.status_code == 200
        experiments = response.json()
        assert isinstance(experiments, list)

    def test_get_experiment(self):
        # First create an experiment
        config = {
            "workload_type": "steady",
            "dataset_id": "synthetic",
            "trace_window_minutes": 5,
            "policy": "fixed",
            "initial_replicas": 2,
            "min_replicas": 1,
            "max_replicas": 4,
            "replica_startup_delay": 5,
            "scaling_cooldown": 10,
            "decision_interval": 2,
            "episode_duration": 30,
            "sla_threshold": 30,
            "random_seed": 42,
            "reward_alpha": 0.1,
            "reward_beta": 1.0,
            "reward_gamma": 0.01,
            "reward_delta": 0.01,
            "ppo_timesteps": 1000,
            "ppo_alpha": 0.1,
            "ppo_beta": 0.01,
            "ppo_gamma": 0.99,
            "short_run": True,
        }
        
        create_resp = client.post("/api/experiments", json=config)
        exp_id = create_resp.json()["id"]
        
        # Get the experiment
        response = client.get(f"/api/experiments/{exp_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == exp_id

    def test_cancel_experiment(self):
        # Test that cancel returns 400 for already completed experiment
        config = {
            "workload_type": "steady",
            "dataset_id": "synthetic",
            "trace_window_minutes": 1,
            "policy": "threshold",
            "initial_replicas": 2,
            "min_replicas": 1,
            "max_replicas": 4,
            "replica_startup_delay": 5,
            "scaling_cooldown": 10,
            "decision_interval": 2,
            "episode_duration": 10,
            "sla_threshold": 30,
            "random_seed": 42,
            "reward_alpha": 0.1,
            "reward_beta": 1.0,
            "reward_gamma": 0.01,
            "reward_delta": 0.01,
            "ppo_timesteps": 1000,
            "ppo_alpha": 0.1,
            "ppo_beta": 0.01,
            "ppo_gamma": 0.99,
            "short_run": True,
        }
        
        create_resp = client.post("/api/experiments", json=config)
        assert create_resp.status_code == 200
        exp_id = create_resp.json()["id"]
        
        # Wait for completion
        import time
        for _ in range(10):
            time.sleep(1)
            get_resp = client.get(f"/api/experiments/{exp_id}")
            if get_resp.json()["state"] in ["completed", "failed", "cancelled"]:
                break
        
        # Try to cancel completed experiment - should fail
        response = client.post(f"/api/experiments/{exp_id}/cancel")
        assert response.status_code == 400

    def test_get_metrics_not_found(self):
        response = client.get("/api/experiments/nonexistent/metrics")
        assert response.status_code == 404

    def test_get_timeline_not_found(self):
        response = client.get("/api/experiments/nonexistent/timeline")
        assert response.status_code == 404

    def test_export_experiment_json(self):
        config = {
            "workload_type": "steady",
            "dataset_id": "synthetic",
            "trace_window_minutes": 1,
            "policy": "fixed",
            "initial_replicas": 2,
            "min_replicas": 1,
            "max_replicas": 4,
            "replica_startup_delay": 5,
            "scaling_cooldown": 10,
            "decision_interval": 2,
            "episode_duration": 10,
            "sla_threshold": 30,
            "random_seed": 42,
            "reward_alpha": 0.1,
            "reward_beta": 1.0,
            "reward_gamma": 0.01,
            "reward_delta": 0.01,
            "ppo_timesteps": 1000,
            "ppo_alpha": 0.1,
            "ppo_beta": 0.01,
            "ppo_gamma": 0.99,
            "short_run": True,
        }
        
        create_resp = client.post("/api/experiments", json=config)
        exp_id = create_resp.json()["id"]
        
        # Wait for completion (short run)
        import time
        for _ in range(10):
            time.sleep(1)
            get_resp = client.get(f"/api/experiments/{exp_id}")
            if get_resp.json()["state"] == "completed":
                break
        
        response = client.get(f"/api/experiments/{exp_id}/export?format=json")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]


class TestTrainingEndpoints:
    """Tests for training endpoints."""

    def test_start_training(self):
        config = {
            "timesteps": 1000,
            "n_steps": 64,
            "batch_size": 16,
            "n_epochs": 2,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "hidden_units": 64,
            "reward_alpha": 0.1,
            "reward_beta": 1.0,
            "reward_gamma": 0.01,
            "reward_delta": 0.01,
            "dataset_id": "synthetic",
            "workload_type": "steady",
            "short_run": True,
        }
        
        response = client.post("/api/training", json=config)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["state"] == "queued"

    def test_list_models(self):
        response = client.get("/api/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)


class TestValidation:
    """Tests for input validation."""

    def test_invalid_experiment_config(self):
        config = {
            "workload_type": "invalid_type",
            "dataset_id": "synthetic",
            "policy": "threshold",
        }
        
        response = client.post("/api/experiments", json=config)
        assert response.status_code == 422  # Validation error

    def test_invalid_policy(self):
        config = {
            "workload_type": "steady",
            "dataset_id": "synthetic",
            "policy": "invalid_policy",
            "initial_replicas": 2,
            "min_replicas": 1,
            "max_replicas": 4,
            "replica_startup_delay": 5,
            "scaling_cooldown": 10,
            "decision_interval": 2,
            "episode_duration": 30,
            "sla_threshold": 30,
            "random_seed": 42,
            "reward_alpha": 0.1,
            "reward_beta": 1.0,
            "reward_gamma": 0.01,
            "reward_delta": 0.01,
            "ppo_timesteps": 1000,
            "ppo_alpha": 0.1,
            "ppo_beta": 0.01,
            "ppo_gamma": 0.99,
            "short_run": True,
        }
        
        response = client.post("/api/experiments", json=config)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])