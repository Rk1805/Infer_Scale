"""Portable checkpoint and independent workload evidence for the RL study."""

import numpy as np
from stable_baselines3 import PPO

from inferscale.experiments.rl_study import (
    CONFIG, REWARD, TEST_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS,
    TrainingEnv, workload,
)
from inferscale.policies.ppo import PPOPolicy
from inferscale.rl.env import InferScaleEnv
from inferscale.rl.features import ScaledObservations


def test_workloads_are_reproducible_and_test_traffic_is_distinct():
    assert not set(TRAIN_SEEDS) & set(VALIDATION_SEEDS + TEST_SEEDS)
    assert not set(VALIDATION_SEEDS) & set(TEST_SEEDS)
    first = workload('bursty', TEST_SEEDS[0])
    assert first == workload('bursty', TEST_SEEDS[0])
    second = workload('bursty', TEST_SEEDS[1])
    assert [r.arrival_time_seconds for r in first] != [r.arrival_time_seconds for r in second]
    assert all(10 < r.arrival_time_seconds < 170 for r in first)
    assert len({r.id for r in first}) == len(first)


def test_training_resets_change_workload_without_test_data():
    env = TrainingEnv()
    env.reset()
    first = [r.id for r in env.simulator.requests]
    env.reset()
    second = [r.id for r in env.simulator.requests]
    assert first != second
    assert first[0].startswith('bursty-100-')
    assert second[0].startswith('steady-101-')


def test_checkpoint_retains_input_scaling_and_loads_through_existing_policy(tmp_path):
    env = InferScaleEnv(workload('steady', 100), config=CONFIG, reward_config=REWARD)
    model = PPO('MlpPolicy', env, seed=7, n_steps=8, batch_size=8, n_epochs=1,
                policy_kwargs={'net_arch': [16, 16],
                               'features_extractor_class': ScaledObservations})
    model.learn(8)
    path = tmp_path / 'ppo.zip'
    model.save(path)
    loaded = PPOPolicy({'model_path': str(path), 'device': 'cpu'})
    assert loaded.model is not None
    obs, _ = env.reset()
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                  loaded.model.predict(obs, deterministic=True)[0])
    assert loaded.model.policy.features_extractor.scales.shape == (12,)
    # Simulated serving must use the same saved transformations as training.
    assert loaded.act(env.simulator.get_observation(), env.simulator) == int(
        model.predict(obs, deterministic=True)[0])
