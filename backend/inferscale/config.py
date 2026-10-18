"""Global configuration for InferScale."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="INFERSCALE_", env_file=".env", extra="ignore")

    # Database
    db_url: str = "sqlite:///./experiments.db"

    # Simulator
    startup_delay: float = 10.0
    cooldown: float = 30.0
    min_replicas: int = 1
    max_replicas: int = 8
    initial_replicas: int = 2
    replica_capacity: int = 10
    decision_interval: float = 5.0
    episode_duration: float = 3600.0
    sla_threshold: float = 30.0
    random_seed: int = 42

    # Policies
    fixed_capacity: int = 4
    threshold_upper: float = 0.75
    threshold_lower: float = 0.25
    scaling_in_duration: float = 60.0
    queue_based_target: float = 0.5

    # RL
    ppo_timesteps: int = 200000
    ppo_alpha: float = 0.1
    ppo_beta: float = 0.01
    ppo_gamma: float = 0.99
    ppo_delta: float = 0.01
    ppo_hidden_units: int = 128
    ppo_n_steps: int = 512
    ppo_n_epochs: int = 10
    ppo_batch_size: int = 64
    ppo_learning_rate: float = 3e-4

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Paths
    data_dir: str = "./data"
    models_dir: str = "./models"
    experiments_dir: str = "./experiments"
    reports_dir: str = "./reports"

    # Cost model
    gpu_cost_per_replica_second: float = 0.01
    cost_normalization: float = 1.0


settings = Settings()
