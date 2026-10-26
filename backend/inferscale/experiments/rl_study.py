"""Reproduce the assignment's seeded PPO training and held-out policy study.

python -m inferscale.experiments.rl_study train
python -m inferscale.experiments.rl_study evaluate
All learning/selection uses training/validation only. Test traces are generated
only by the separate evaluate command, after the selected checkpoint is frozen.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from inferscale.experiments.benchmark import code_provenance, extract_metrics, trace_digest
from inferscale.policies import get_policy
from inferscale.rl.env import InferScaleEnv
from inferscale.rl.features import ScaledObservations
from inferscale.schemas.request import RequestSchema
from inferscale.simulator.engine import SimulatorEngine

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / 'artifacts' / 'rl_study'
CONFIG = dict(initial_replicas=2, min_replicas=1, max_replicas=8,
              startup_delay=5., cooldown=10., decision_interval=5.,
              episode_duration=180., sla_threshold=2., replica_capacity=2,
              cost_per_replica_second=0.0025, slo_target=0.95)
REWARD = dict(alpha=0.1, beta=1., gamma=0.01, delta=0.01)
TRAIN_SEEDS = list(range(100, 200))
VALIDATION_SEEDS = [501, 502, 503]
TEST_SEEDS = [1001, 1002, 1003, 1004, 1005]
PROFILES = dict(bursty=[2., 12., 3., 20.], steady=[8., 8., 8., 8.],
                shifted=[15., 2., 22., 4.])
POLICIES = ['fixed', 'threshold', 'queue', 'ppo']


def workload(profile: str, seed: int) -> list[RequestSchema]:
    """Seeded Poisson arrivals and tokens; 10 s warmup/drain, four 40 s phases.

    Arrival rates also vary by a seeded 0.8--1.2 multiplier per phase. Synthetic
    labels are explicit and never imply actual Azure or BurstGPT provenance.
    """
    rng = random.Random(seed)
    requests = []
    for phase, rate in enumerate(PROFILES[profile]):
        start = 10 + phase * 40
        end = start + 40
        rate *= rng.uniform(0.8, 1.2)
        arrival = start + rng.expovariate(rate)
        while arrival < end:
            requests.append(RequestSchema(
                id=f'{profile}-{seed}-{len(requests)}', arrival_time_seconds=arrival,
                prompt_tokens=rng.randint(100, 600), output_tokens=rng.randint(50, 350),
                source='synthetic_rl_study',
            ))
            arrival += rng.expovariate(rate)
    return requests


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TrainingEnv(InferScaleEnv):
    """Rotate disjoint training traces at reset without modifying the base env."""

    def __init__(self):
        self.episode_index = 0
        super().__init__(requests=workload('bursty', TRAIN_SEEDS[0]),
                         config=CONFIG, reward_config=REWARD)

    def reset(self, **kwargs):
        seed = TRAIN_SEEDS[self.episode_index % len(TRAIN_SEEDS)]
        profile = 'bursty' if self.episode_index % 2 == 0 else 'steady'
        self.simulator.requests = workload(profile, seed)
        self.episode_index += 1
        return super().reset(**kwargs)


def validation(model):
    returns = []
    for profile in ('bursty', 'steady'):
        for seed in VALIDATION_SEEDS:
            env = InferScaleEnv(workload(profile, seed), config=CONFIG, reward_config=REWARD)
            obs, _ = env.reset(seed=seed)
            total = 0.
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(int(action))
                total += reward
                if done or truncated:
                    break
            returns.append(total)
            env.close()
    return float(np.mean(returns))


class StudyCallback(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.validation_rows = []
        self.episodes = []
        self.best_reward = -float('inf')
        self.best_step = 0
        self.started = time.perf_counter()

    def checkpoint(self):
        score = validation(self.model)
        self.validation_rows.append(dict(timesteps=self.num_timesteps,
                                         mean_validation_return=score))
        # Record the untrained control, but only trained policies are eligible.
        if self.num_timesteps > 0 and score > self.best_reward:
            self.best_reward, self.best_step = score, self.num_timesteps
            self.model.save(self.directory / 'ppo_autoscaler.zip')
        write_csv(self.directory / 'validation.csv', self.validation_rows)
        print(f'Validation step={self.num_timesteps}, return={score:.4f}, '
              f'best={self.best_reward:.4f}, elapsed={time.perf_counter()-self.started:.1f}s',
              flush=True)

    def _on_training_start(self):
        self.checkpoint()

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episodes.append(dict(timesteps=self.num_timesteps,
                                          episode_return=float(info['episode']['r']),
                                          episode_length=int(info['episode']['l'])))
        if self.num_timesteps % 2048 == 0:
            recent = [row['episode_return'] for row in self.episodes[-20:]]
            print(f'Train step={self.num_timesteps}, episodes={len(self.episodes)}, '
                  f'recent_return={np.mean(recent):.4f}', flush=True)
            if self.episodes:
                write_csv(self.directory / 'training.csv', self.episodes)
        if self.num_timesteps % 4096 == 0:
            self.checkpoint()
        return True


def train(args):
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'training_manifest.json').exists():
        raise SystemExit('Training artifacts already exist; choose a new --directory.')
    torch.set_num_threads(1)
    env = TrainingEnv()
    check_env(env, warn=True)
    env.episode_index = 0
    model = PPO('MlpPolicy', Monitor(env), device='cpu', seed=7, verbose=0,
                learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10, gamma=0.99,
                ent_coef=args.entropy_coef,
                policy_kwargs=dict(net_arch=[128, 128],
                                   features_extractor_class=ScaledObservations))
    callback = StudyCallback(directory)
    started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    if callback.validation_rows[-1]['timesteps'] != model.num_timesteps:
        callback.checkpoint()
    model.save(directory / 'ppo_final.zip')
    write_csv(directory / 'training.csv', callback.episodes)
    manifest = dict(
        config=CONFIG, reward=REWARD, training_seeds=TRAIN_SEEDS,
        validation_seeds=VALIDATION_SEEDS, test_seeds=TEST_SEEDS,
        training_profiles=['bursty', 'steady'], validation_profiles=['bursty', 'steady'],
        test_profiles=['bursty', 'steady', 'shifted'], rates=PROFILES,
        ppo_seed=7, training_steps=int(model.num_timesteps),
        selected_step=callback.best_step, validation_return=callback.best_reward,
        selection='Maximum mean deterministic validation reward among trained checkpoints; no test selection',
        elapsed_seconds=time.perf_counter()-started,
        checkpoint_sha256=hashlib.sha256((directory / 'ppo_autoscaler.zip').read_bytes()).hexdigest(),
        provenance=code_provenance(), python=platform.python_version(),
        versions={name: importlib.metadata.version(name)
                  for name in ['torch', 'stable-baselines3', 'gymnasium', 'numpy']},
        policy=dict(learning_rate=3e-4, n_steps=512, batch_size=64, ent_coef=args.entropy_coef,
                    n_epochs=10, gamma=0.99, net_arch=[128, 128],
                    features_extractor='inferscale.rl.features.ScaledObservations'),
    )
    dump(directory / 'training_manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def run_policy(requests, policy, seed):
    engine = SimulatorEngine([request.model_copy(deep=True) for request in requests],
                             seed=seed, **CONFIG)
    timeline = []
    while engine.events and engine.current_time < engine.episode_duration:
        event = engine.step()
        if event and event.get('decision_point'):
            obs = {**event['observation'], 'time': engine.current_time}
            action = int(policy.act(obs, engine))
            timeline.append(dict(time=engine.current_time, action=action,
                                 active=obs['active_replicas'], starting=obs['starting_replicas'],
                                 queue=obs['queue_length']))
            engine.take_action(action)
    return extract_metrics(engine), timeline


def select(args):
    """Choose/record candidates using validation only, before tests are opened."""
    torch.set_num_threads(1)
    directory = args.directory
    if (directory / 'test_evaluation.json').exists():
        raise SystemExit('Test results already exist: use a new study for further tuning.')
    candidates = []
    files = ['training_manifest.json', 'training.csv', 'validation.csv',
             'ppo_autoscaler.zip', 'ppo_final.zip']
    for source in args.candidates:
        manifest = json.loads((source / 'training_manifest.json').read_text())
        if manifest['selected_step'] <= 0 or manifest['config'] != CONFIG:
            raise ValueError('Candidate must be trained under the shared environment settings.')
        checkpoint = source / 'ppo_autoscaler.zip'
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
            raise ValueError('Candidate checkpoint hash differs from its manifest.')
        # Record the actual saved library setting, including older candidate manifests.
        manifest['policy']['ent_coef'] = float(PPO.load(checkpoint, device='cpu').ent_coef)
        archive = directory / 'candidates' / source.name
        archive.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copy2(source / name, archive / name)
        dump(archive / 'training_manifest.json', manifest)
        candidates.append(dict(directory=str(source), archive=archive,
                               validation_return=manifest['validation_return'],
                               selected_step=manifest['selected_step'],
                               entropy_coef=manifest['policy']['ent_coef']))
    selected = max(candidates, key=lambda row: row['validation_return'])
    for name in files:
        shutil.copy2(selected['archive'] / name, directory / name)
    dump(directory / 'selection.json', dict(
        criterion='Highest mean deterministic validation return; earlier candidate breaks ties',
        test_used_for_selection=False,
        selected_candidate=selected['directory'],
        candidates=[{key: str(value) if isinstance(value, Path) else value
                     for key, value in row.items()} for row in candidates],
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for subdir in ('rl', 'simulator', 'policies', 'experiments')
                       for path in (ROOT / 'backend' / 'inferscale' / subdir).glob('*.py')},
    ))
    print(f'Selected {selected["directory"]}: validation={selected["validation_return"]:.5f}',
          flush=True)


def evaluate(args):
    torch.set_num_threads(1)
    directory = args.directory
    manifest = json.loads((directory / 'training_manifest.json').read_text())
    if manifest['selected_step'] <= 0:
        raise SystemExit('Evaluation requires a trained checkpoint.')
    checkpoint = directory / 'ppo_autoscaler.zip'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != manifest['checkpoint_sha256'] or manifest['config'] != CONFIG:
        raise SystemExit('Checkpoint or protocol differs from training manifest.')
    test_seeds = TEST_SEEDS if args.split == 'test' else VALIDATION_SEEDS
    profiles = ['bursty', 'steady', 'shifted'] if args.split == 'test' else ['bursty', 'steady']
    assert not set(TRAIN_SEEDS) & set(test_seeds)
    runs = []
    traces = directory / f'{args.split}_traces'
    traces.mkdir(exist_ok=True)
    for profile in profiles:
        for seed in test_seeds:
            requests = workload(profile, seed)
            dump(traces / f'{profile}_{seed}.json', [req.model_dump() for req in requests])
            trace_hash = trace_digest(requests)
            for name in POLICIES:
                policy = get_policy(name, dict(model_path=str(checkpoint), device='cpu',
                                               fixed_capacity=4, cooldown=CONFIG['cooldown']))
                if name == 'ppo' and policy.model is None:
                    raise RuntimeError('Checkpoint unavailable; fallback evaluation forbidden.')
                metrics, timeline = run_policy(requests, policy, seed)
                row = dict(profile=profile, seed=seed, policy=name,
                           trace_hash=trace_hash, metrics=metrics)
                runs.append(row)
                if seed == test_seeds[0]:
                    dump(directory / f'{args.split}_{profile}_{name}_timeline.json', timeline)
                print(f'{args.split} {profile}/{seed}/{name}: cost={metrics["total_cost"]:.4f}, '
                      f'on_time={metrics["slo_attainment_lower_bound"]:.3f}, '
                      f'p95_ms={metrics["p95_latency_ms"]:.1f}', flush=True)
    dump(directory / f'{args.split}_evaluation.json',
         dict(split=args.split, config=CONFIG, checkpoint_sha256=digest,
              training_manifest=manifest, profiles=profiles, seeds=test_seeds, runs=runs,
              provenance=code_provenance(),
              limitations=['Synthetic analytical simulation; no measured GPU results',
                           'One PPO training seed; test seeds vary requests, not learned agents',
                           'Shifted workload was not used in training or checkpoint selection']))
    flat = [dict(profile=row['profile'], seed=row['seed'], policy=row['policy'],
                 trace_hash=row['trace_hash'], **row['metrics']) for row in runs]
    write_csv(directory / f'{args.split}_runs.csv', flat)
    print(f'Saved {len(runs)} paired runs.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'select', 'evaluate'])
    parser.add_argument('--directory', type=Path, default=ARTIFACTS)
    parser.add_argument('--timesteps', type=int, default=32768)
    parser.add_argument('--entropy-coef', type=float, default=0.02)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--candidates', type=Path, nargs='+',
                        default=[ARTIFACTS, ROOT / 'artifacts' / 'rl_study_entropy'])
    args = parser.parse_args()
    if args.timesteps < 512:
        parser.error('--timesteps must be at least 512')
    if not np.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        parser.error('--entropy-coef must be finite and nonnegative')
    {'train': train, 'select': select, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
