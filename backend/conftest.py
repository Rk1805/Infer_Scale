"""Pytest configuration and fixtures."""

import pytest
import sys
import atexit
import os
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Package imports initialize a global database. Isolate it before test collection,
# including the legacy API tests, so the contributor's database is never modified.
_test_storage = tempfile.TemporaryDirectory(prefix="inferscale-tests-")
atexit.register(_test_storage.cleanup)
os.environ["INFERSCALE_DB_URL"] = "sqlite:///" + str(Path(_test_storage.name) / "experiments.db")
os.environ["INFERSCALE_REPORTS_DIR"] = str(Path(_test_storage.name) / "reports")


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
