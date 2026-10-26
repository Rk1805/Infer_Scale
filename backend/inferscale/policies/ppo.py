"""PPO-based RL policy."""

import os
from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from inferscale.policies.base import Policy


class PPOPolicy(Policy):
    """PPO-based autoscaling policy that loads a trained model."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.model_path = self.config.get("model_path", None)
        self.model = None
        if self.model_path and os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path, device=self.config.get("device", "auto"))

    def act(self, observation: Dict[str, Any], engine) -> int:
        """Use trained PPO model to select action."""
        if self.model is None:
            # Fallback to threshold behavior
            active = observation.get("active_replicas", 0)
            starting = observation.get("starting_replicas", 0)
            total = active + starting
            utilization = observation.get("utilization", 0.0)
            if utilization > 0.75 and total < engine.max_replicas:
                return 2
            if utilization < 0.25 and total > engine.min_replicas:
                return 0
            return 1

        obs_array = self._observation_to_array(observation)
        action, _states = self.model.predict(obs_array, deterministic=True)
        return int(action)

    def _observation_to_array(self, obs: Dict[str, Any]) -> np.ndarray:
        """Convert observation dict to array for PPO model."""
        keys = [
            "queue_length", "estimated_work", "arrival_rate",
            "avg_prompt_tokens", "avg_output_tokens",
            "active_replicas", "total_replicas", "utilization",
            "p95_latency", "sla_violation_rate",
            "cooldown_remaining", "starting_replicas",
        ]
        return np.array([float(obs.get(k, 0.0)) for k in keys])

    def get_policy_config(self) -> Dict[str, Any]:
        return {
            "policy": "ppo",
            "model_path": self.model_path,
        }

    def create(self, config: Dict[str, Any]) -> "PPOPolicy":
        return PPOPolicy(config)


def train_ppo(env, total_timesteps: int = 200000, **kwargs) -> PPO:
    """Train a PPO model on the given environment."""
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=kwargs.get("n_steps", 512),
        batch_size=kwargs.get("batch_size", 64),
        n_epochs=kwargs.get("n_epochs", 10),
        learning_rate=kwargs.get("learning_rate", 3e-4),
        gamma=kwargs.get("gamma", 0.99),
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def save_ppo_model(model: PPO, path: str) -> None:
    """Save a trained PPO model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def evaluate_ppo_model(model_path: str, env) -> Dict[str, Any]:
    """Evaluate a trained PPO model."""
    model = PPO.load(model_path)
    obs, info = env.reset()
    episode_rewards = []
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            done = True
            episode_rewards.append(total_reward)
    return {"avg_reward": total_reward / len(episode_rewards), "rewards": episode_rewards}
