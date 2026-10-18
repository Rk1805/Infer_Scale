"""Gymnasium environment for InferScale simulation."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from inferscale.simulator.engine import SimulatorEngine
from inferscale.rl.reward import RewardFunction


class InferScaleEnv(gym.Env):
    """Gymnasium environment for LLM inference autoscaling."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        requests: Optional[List] = None,
        config: Optional[Dict] = None,
        reward_config: Optional[Dict] = None,
        max_episode_steps: int = 1000,
    ):
        super().__init__()

        self.config = config or {}
        self.max_episode_steps = max_episode_steps

        # Initialize simulator
        self.simulator = SimulatorEngine(
            requests=requests or [],
            initial_replicas=self.config.get("initial_replicas", 2),
            min_replicas=self.config.get("min_replicas", 1),
            max_replicas=self.config.get("max_replicas", 8),
            startup_delay=self.config.get("startup_delay", 10.0),
            cooldown=self.config.get("cooldown", 30.0),
            decision_interval=self.config.get("decision_interval", 5.0),
            episode_duration=self.config.get("episode_duration", 3600.0),
            sla_threshold=self.config.get("sla_threshold", 30.0),
            replica_capacity=self.config.get("replica_capacity", 10),
            seed=self.config.get("random_seed", 42),
        )

        # Initialize reward function
        self.reward_fn = RewardFunction(reward_config)

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: scale down, 1: hold, 2: scale up

        # Observation space: normalized values in [0, 1] or [-inf, inf]
        # 12 features: queue_length, estimated_work, arrival_rate, avg_prompt_tokens,
        # avg_output_tokens, active_replicas, total_replicas, utilization,
        # p95_latency, sla_violation_rate, cooldown_remaining, starting_replicas
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([100, 10000, 100, 1000, 1000, 100, 100, 1, 100, 1, 100, 100], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self.current_step = 0
        self.last_obs = None
        self.total_reward = 0.0
        self.previous_replicas = self.config.get("initial_replicas", 2)
        self.render_mode = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            self.simulator.rng = np.random.default_rng(seed)

        self.simulator.reset()
        self.reward_fn.reset()
        self.current_step = 0
        self.total_reward = 0.0
        self.previous_replicas = self.simulator.initial_replicas

        obs = self._get_observation()
        info = {"episode_start": True}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.current_step += 1

        # Apply action
        self.simulator.take_action(action)

        # Run simulator until next decision point or episode end
        results = self.simulator.run_until(
            self.simulator.current_time + self.config.get("decision_interval", 5.0)
        )

        # Extract metrics from results
        cost = 0.0
        sla_violated = False
        queue_pressure = 0.0
        scaling_churn = 0

        if results:
            latest = results[-1]
            if "metrics" in latest:
                metrics = latest["metrics"]
                cost = self._extract_cost_from_metrics(metrics)
                sla_violated = metrics.get("sla_violation_rate", 0) > 0.5  # heuristic
                queue_pressure = self._extract_queue_pressure_from_metrics(metrics)
                scaling_churn = self._extract_scaling_churn_from_metrics(metrics)

        # Compute reward
        reward_dict = self.reward_fn.compute(
            cost=cost,
            sla_violation=sla_violated,
            queue_pressure=queue_pressure,
            scaling_churn=scaling_churn,
        )
        reward = reward_dict["total"]

        self.total_reward += reward
        self.previous_replicas = self.simulator.get_current_replicas()
        self.last_obs = self._get_observation()

        # Check if episode is done
        terminated = self.simulator.current_time >= self.simulator.episode_duration
        truncated = self.current_step >= self.max_episode_steps

        info = {
            "reward_components": reward_dict,
            "simulator_metrics": self.simulator.get_metrics() if hasattr(self.simulator, 'get_metrics') else {},
            "step": self.current_step,
            "time": self.simulator.current_time,
        }

        return self.last_obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation from simulator."""
        obs_dict = self.simulator.get_observation()
        obs_array = np.array([
            obs_dict.get("queue_length", 0),
            obs_dict.get("estimated_work", 0),
            obs_dict.get("arrival_rate", 0),
            obs_dict.get("avg_prompt_tokens", 0),
            obs_dict.get("avg_output_tokens", 0),
            obs_dict.get("active_replicas", 0),
            obs_dict.get("total_replicas", 0),
            obs_dict.get("utilization", 0),
            obs_dict.get("p95_latency", 0),
            obs_dict.get("sla_violation_rate", 0),
            obs_dict.get("cooldown_remaining", 0),
            obs_dict.get("starting_replicas", 0),
        ], dtype=np.float32)
        return obs_array

    def _extract_cost_from_metrics(self, metrics: Dict) -> float:
        """Extract cost from simulator metrics."""
        return metrics.get("total_cost", 0.0) * 100  # scale for observation

    def _extract_queue_pressure_from_metrics(self, metrics: Dict) -> float:
        """Extract queue pressure from simulator metrics."""
        queue_len = metrics.get("avg_queue_depth", 0)
        active_reps = metrics.get("avg_active_replicas", 1)
        return queue_len / max(1, active_reps)

    def _extract_scaling_churn_from_metrics(self, metrics: Dict) -> int:
        """Extract scaling churn from simulator metrics."""
        return metrics.get("scaling_events", 0)

    def render(self):
        """Render the environment."""
        pass

    def close(self):
        """Clean up resources."""
        pass


def make_inferscale_env(requests: List, config: Dict, reward_config: Dict):
    """Factory function to create InferScale environment."""
    return InferScaleEnv(requests=requests, config=config, reward_config=reward_config)
