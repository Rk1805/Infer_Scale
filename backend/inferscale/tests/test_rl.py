"""Tests for RL environment and reward function."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from inferscale.rl.env import InferScaleEnv, make_inferscale_env
from inferscale.rl.reward import RewardFunction, compute_normalized_cost, compute_sla_violation, compute_queue_pressure, compute_scaling_churn
from inferscale.schemas.request import RequestSchema
from inferscale.simulator.engine import SimulatorEngine


@pytest.fixture
def sample_requests():
    requests = []
    for i in range(50):
        requests.append(RequestSchema(
            id=f"req_{i}",
            arrival_time_seconds=i * 0.2,
            prompt_tokens=50,
            output_tokens=30,
            source="test",
        ))
    return requests


@pytest.fixture
def env_config():
    return {
        "initial_replicas": 2,
        "min_replicas": 1,
        "max_replicas": 4,
        "startup_delay": 5.0,
        "cooldown": 10.0,
        "decision_interval": 2.0,
        "episode_duration": 20.0,
        "sla_threshold": 5.0,
        "replica_capacity": 5,
        "random_seed": 42,
    }


@pytest.fixture
def reward_config():
    return {
        "alpha": 0.1,
        "beta": 1.0,
        "gamma": 0.01,
        "delta": 0.01,
    }


class TestRewardFunction:
    """Tests for the reward function."""

    def test_compute_reward_components(self):
        reward_fn = RewardFunction({"alpha": 0.1, "beta": 1.0, "gamma": 0.01, "delta": 0.01})
        
        result = reward_fn.compute(
            cost=5.0,
            sla_violation=False,
            queue_pressure=2.0,
            scaling_churn=1,
        )
        
        assert "total" in result
        assert "cost" in result
        assert "sla" in result
        assert "queue" in result
        assert "churn" in result
        assert result["total"] < 0  # Negative reward (penalty)
        assert result["sla"] == 0  # No SLA violation
        assert result["cost"] < 0  # Cost penalty

    def test_sla_violation_penalty(self):
        reward_fn = RewardFunction({"beta": 1.0})
        
        result_no_violation = reward_fn.compute(cost=0, sla_violation=False, queue_pressure=0, scaling_churn=0)
        result_violation = reward_fn.compute(cost=0, sla_violation=True, queue_pressure=0, scaling_churn=0)
        
        assert result_violation["sla"] == -1.0
        assert result_violation["total"] < result_no_violation["total"]

    def test_cost_normalization(self):
        reward_fn = RewardFunction({"alpha": 0.1})
        
        result_low = reward_fn.compute(cost=1.0, sla_violation=False, queue_pressure=0, scaling_churn=0)
        result_high = reward_fn.compute(cost=10.0, sla_violation=False, queue_pressure=0, scaling_churn=0)
        result_very_high = reward_fn.compute(cost=100.0, sla_violation=False, queue_pressure=0, scaling_churn=0)
        
        # Cost should be capped at 10x
        assert result_high["cost"] == result_very_high["cost"]
        assert result_low["cost"] > result_high["cost"]  # Less negative = better

    def test_queue_pressure_normalization(self):
        reward_fn = RewardFunction({"gamma": 0.01})
        
        result = reward_fn.compute(cost=0, sla_violation=False, queue_pressure=10.0, scaling_churn=0)
        # Queue pressure capped at 5x
        assert result["queue"] == -0.01 * (5.0 / 5.0)

    def test_scaling_churn_normalization(self):
        reward_fn = RewardFunction({"delta": 0.01})
        
        result = reward_fn.compute(cost=0, sla_violation=False, queue_pressure=0, scaling_churn=10)
        # Churn capped at 3
        assert result["churn"] == -0.01 * (3.0 / 3.0)

    def test_finite_rewards(self):
        reward_fn = RewardFunction()
        
        # Extreme values should not produce infinite rewards
        result = reward_fn.compute(
            cost=float('inf'),
            sla_violation=True,
            queue_pressure=float('inf'),
            scaling_churn=1000,
        )
        
        assert np.isfinite(result["total"])

    def test_weight_changes(self):
        # Changing weights should affect reward
        reward_fn1 = RewardFunction({"alpha": 0.1, "beta": 0.01})
        reward_fn2 = RewardFunction({"alpha": 1.0, "beta": 0.01})
        
        result1 = reward_fn1.compute(cost=5.0, sla_violation=False, queue_pressure=0, scaling_churn=0)
        result2 = reward_fn2.compute(cost=5.0, sla_violation=False, queue_pressure=0, scaling_churn=0)
        
        assert result1["cost"] != result2["cost"]


class TestRewardHelpers:
    """Tests for reward helper functions."""

    def test_compute_normalized_cost(self):
        cost = compute_normalized_cost(active_replicas=4, max_replicas=8, delta_time=10.0, cost_per_replica_second=0.01)
        expected = (4 * 10.0 * 0.01) / (8 * 10.0 * 0.01)
        assert cost == expected

    def test_compute_sla_violation(self):
        assert compute_sla_violation(100.0, 50.0, 30.0) == True   # 50s > 30s threshold
        assert compute_sla_violation(70.0, 50.0, 30.0) == False  # 20s < 30s threshold

    def test_compute_queue_pressure(self):
        pressure = compute_queue_pressure(queue_length=10, active_replicas=2)
        assert pressure == 5.0
        
        pressure = compute_queue_pressure(queue_length=0, active_replicas=2)
        assert pressure == 0.0

    def test_compute_scaling_churn(self):
        assert compute_scaling_churn(2, 3) == 1
        assert compute_scaling_churn(3, 2) == 1
        assert compute_scaling_churn(2, 2) == 0
        assert compute_scaling_churn(1, 4) == 3


class TestInferScaleEnv:
    """Tests for the Gymnasium environment."""

    def test_env_creation(self, sample_requests, env_config, reward_config):
        env = InferScaleEnv(requests=sample_requests, config=env_config, reward_config=reward_config)
        
        assert env.action_space.n == 3
        assert env.observation_space.shape == (12,)
        assert env.simulator is not None

    def test_reset(self, sample_requests, env_config, reward_config):
        env = InferScaleEnv(requests=sample_requests, config=env_config, reward_config=reward_config)
        
        obs, info = env.reset(seed=42)
        
        assert obs.shape == (12,)
        assert isinstance(info, dict)
        assert info.get("episode_start") == True

    def test_step(self, sample_requests, env_config, reward_config):
        env = InferScaleEnv(requests=sample_requests, config=env_config, reward_config=reward_config)
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(1)  # hold action
        
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "reward_components" in info

    def test_action_space_validity(self, sample_requests, env_config, reward_config):
        env = InferScaleEnv(requests=sample_requests, config=env_config, reward_config=reward_config)
        env.reset()
        
        # All actions should be valid
        for action in [0, 1, 2]:
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)

    def test_observation_space_bounds(self, sample_requests, env_config, reward_config):
        env = InferScaleEnv(requests=sample_requests, config=env_config, reward_config=reward_config)
        env.reset()
        
        # Run several steps and check observations stay in bounds
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(1)
            assert np.all(obs >= env.observation_space.low)
            assert np.all(obs <= env.observation_space.high)
            if terminated or truncated:
                break

    def test_episode_termination(self, sample_requests, env_config, reward_config):
        env = InferScaleEnv(requests=sample_requests, config=env_config, reward_config=reward_config, max_episode_steps=5)
        env.reset()
        
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(1)
            steps += 1
            if steps > 100:
                break
        
        # Should terminate due to max_episode_steps
        assert truncated or terminated

    def test_reward_components_in_info(self, sample_requests, env_config, reward_config):
        env = InferScaleEnv(requests=sample_requests, config=env_config, reward_config=reward_config)
        env.reset()
        
        _, _, _, _, info = env.step(1)
        
        assert "reward_components" in info
        components = info["reward_components"]
        assert "total" in components
        assert "cost" in components
        assert "sla" in components
        assert "queue" in components
        assert "churn" in components

    def test_gymnasium_checker(self, sample_requests, env_config, reward_config):
        """Test that environment passes Gymnasium's environment checker."""
        env = InferScaleEnv(requests=sample_requests, config=env_config, reward_config=reward_config)
        
        # This would normally use check_env from stable_baselines3
        # For now, just verify basic interface
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        assert hasattr(env, 'close')

    def test_factory_function(self, sample_requests, env_config, reward_config):
        env = make_inferscale_env(sample_requests, env_config, reward_config)
        assert isinstance(env, InferScaleEnv)


class TestRLIntegration:
    """Integration tests for RL training loop."""

    def test_short_training_run(self, sample_requests, env_config, reward_config):
        """Test that a short training run completes without errors."""
        env = InferScaleEnv(requests=sample_requests, config=env_config, reward_config=reward_config)
        
        from stable_baselines3 import PPO
        
        model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=16, n_epochs=2)
        
        # Very short training
        model.learn(total_timesteps=200)
        
        # Evaluate
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            if terminated or truncated:
                break
        
        assert np.isfinite(total_reward)

    def test_model_save_load(self, sample_requests, env_config, reward_config, tmp_path):
        """Test saving and loading a trained model."""
        env = InferScaleEnv(requests=sample_requests, config=env_config, reward_config=reward_config)
        
        from stable_baselines3 import PPO
        
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=100)
        
        model_path = tmp_path / "test_model.zip"
        model.save(str(model_path))
        
        # Load and verify
        loaded_model = PPO.load(str(model_path))
        
        obs, _ = env.reset()
        action1, _ = model.predict(obs, deterministic=True)
        action2, _ = loaded_model.predict(obs, deterministic=True)
        
        assert action1 == action2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])