"""Synthetic workload generators for testing."""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from inferscale.schemas.request import RequestSchema


def generate_steady_workload(
    num_requests: int = 1000,
    avg_token_count: int = 50,
    arrival_rate: float = 10.0,
    duration_minutes: float = 30.0,
    *,
    rng: random.Random | None = None,
    start_time_seconds: float | None = None,
) -> List[RequestSchema]:
    """Generate a steady-state workload."""
    requests = []
    start_time = datetime.now() - timedelta(minutes=duration_minutes)
    origin = start_time.timestamp() if start_time_seconds is None else start_time_seconds
    rng = rng or random
    for i in range(num_requests):
        t = start_time + timedelta(seconds=i * arrival_rate)
        requests.append(RequestSchema(
            id=f"req_{i}",
            arrival_time_seconds=origin + i * arrival_rate,
            prompt_tokens=rng.randint(20, 200),
            output_tokens=rng.randint(10, 150),
            source="synthetic" if i % 3 == 0 else "azurellm",
        ))
    return requests


def generate_periodic_workload(
    num_requests: int = 2000,
    periods: int = 4,
    period_seconds: int = 60,
    peak_factor: float = 2.0,
    *,
    rng: random.Random | None = None,
    start_time_seconds: float | None = None,
) -> List[RequestSchema]:
    """Generate a periodic (bursty) workload."""
    requests = []
    start_time = datetime.now() - timedelta(hours=periods * period_seconds)
    origin = start_time.timestamp() if start_time_seconds is None else start_time_seconds
    rng = rng or random
    for period in range(periods):
        base_time = start_time + timedelta(seconds=period * period_seconds)
        for i in range(num_requests // periods):
            t = base_time + timedelta(seconds=i * period_seconds)
            multiplier = 1.0 + peak_factor * (period % 3)  # 1x, 1.5x, 2x
            token_count = int(rng.randint(20, 200) * multiplier)
            requests.append(RequestSchema(
                id=f"periodic_{period}_{i}",
                arrival_time_seconds=origin + (period + i) * period_seconds,
                prompt_tokens=token_count,
                output_tokens=rng.randint(10, 150),
                source="synthetic",
            ))
    return requests


def generate_flash_crowd_workload(
    num_requests: int = 5000,
    peak_ratio: float = 5.0,
    *,
    rng: random.Random | None = None,
    start_time_seconds: float | None = None,
) -> List[RequestSchema]:
    """Generate a flash-crowd style workload."""
    requests = []
    start_time = datetime.now() - timedelta(hours=10)
    origin = start_time.timestamp() if start_time_seconds is None else start_time_seconds
    rng = rng or random
    # Simulate a sudden spike
    spike_start = start_time + timedelta(minutes=5)
    spike_end = start_time + timedelta(minutes=15)
    for i in range(num_requests):
        t = start_time + timedelta(seconds=i * 0.1)
        if spike_start <= t <= spike_end:
            multiplier = peak_ratio
        else:
            multiplier = 1.0
        token_count = int(rng.randint(50, 300) * multiplier)
        requests.append(RequestSchema(
            id=f"flash_{i}",
            arrival_time_seconds=origin + i * 0.1,
            prompt_tokens=token_count,
            output_tokens=rng.randint(20, 200),
            source="flashcrowd",
        ))
    return requests


SYNTHETIC_WORKLOADS = {
    "steady": generate_steady_workload,
    "periodic": generate_periodic_workload,
    "flash_crowd": generate_flash_crowd_workload,
}
