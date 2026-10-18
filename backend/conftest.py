"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def sample_requests():
    """Create sample requests for testing."""
    from inferscale.schemas.request import RequestSchema
    requests = []
    for i in range(20):
        requests.append(RequestSchema(
            id=f"req_{i}",
            arrival_time_seconds=i * 0.5,
            prompt_tokens=50,
            output_tokens=30,
            source="test",
        ))
    return requests


@pytest.fixture(scope="session")
def engine_config():
    """Default engine configuration for testing."""
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