"""Regression checks needed before collecting learned-policy evidence."""

import numpy as np
import pytest

from inferscale.rl.env import InferScaleEnv
from inferscale.schemas.request import RequestSchema
from inferscale.simulator.engine import SimulatorEngine


def test_hold_does_not_extend_scaling_cooldown():
    engine = SimulatorEngine([], initial_replicas=1, max_replicas=4,
                             startup_delay=0, cooldown=10, episode_duration=40)
    engine.take_action(2)
    engine.run_until(1)
    assert engine.metrics['scaling_events'] == 1
    for time in (3, 6, 9):
        engine.run_until(time)
        engine.take_action(1)
        engine.run_until(time)
    engine.run_until(10)
    engine.take_action(2)
    engine.run_until(10)
    assert engine.metrics['scaling_events'] == 2
    assert engine.get_current_replicas() == 3


def test_rejected_action_does_not_start_cooldown():
    engine = SimulatorEngine([], initial_replicas=1, max_replicas=2,
                             startup_delay=0, cooldown=10, episode_duration=20)
    engine.run_until(1)
    engine.take_action(0)  # Already at minimum.
    engine.run_until(1)
    engine.take_action(2)
    engine.run_until(1)
    assert engine.metrics['scaling_events'] == 1
    assert engine.get_current_replicas() == 2


def test_completion_matches_one_prefill_and_one_decode_phase():
    request = RequestSchema(id='single', arrival_time_seconds=0,
                            prompt_tokens=100, output_tokens=200)
    engine = SimulatorEngine([request], initial_replicas=1,
                             startup_delay=0, episode_duration=2)
    engine.run_until(2)
    result = engine.completed_requests[0]
    expected = engine.service_model.total_time(100, 200)
    assert result.decode_end_time == pytest.approx(expected)
    assert result.decode_start_time == pytest.approx(result.prefill_end_time)


def test_large_raw_observation_is_in_declared_space():
    env = InferScaleEnv()
    raw = np.zeros(12, dtype=np.float32)
    raw[0], raw[1], raw[8] = 1000, 1e7, 300
    assert env.observation_space.contains(raw)
