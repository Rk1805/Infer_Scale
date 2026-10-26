"""Expose recorded RL evidence and bounded, real checkpoint replays for the UI."""

import csv
import hashlib
import json
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from inferscale.experiments.benchmark import trace_digest
from inferscale.experiments.rl_study import CONFIG, run_policy
from inferscale.experiments.statistics import summarize_metric
from inferscale.policies import get_policy
from inferscale.schemas.request import RequestSchema

ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = ROOT / 'artifacts' / 'rl_study'
router = APIRouter(prefix='/api/rl-study', tags=['rl-study'])
replay_lock = threading.Lock()
Profile = Literal['bursty', 'steady', 'shifted']
POLICIES = ('fixed', 'threshold', 'queue', 'ppo')


def read_json(name):
    try:
        return json.loads((STUDY_ROOT / name).read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise HTTPException(404, 'RL study artifacts are missing. Reproduce the study using the README.')
    except (OSError, ValueError):
        raise HTTPException(503, 'RL study artifacts could not be read.')


def checked_study():
    manifest = read_json('training_manifest.json')
    results = read_json('test_evaluation.json')
    checkpoint = STUDY_ROOT / 'ppo_autoscaler.zip'
    if not checkpoint.is_file():
        raise HTTPException(404, 'The trained PPO checkpoint is missing.')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (digest != manifest.get('checkpoint_sha256')
            or digest != results.get('checkpoint_sha256')):
        raise HTTPException(409, 'Checkpoint and recorded evaluation hashes do not match.')
    if manifest.get('selected_step', 0) <= 0 or results.get('split') != 'test':
        raise HTTPException(409, 'A trained checkpoint and held-out test results are required.')
    return manifest, results, checkpoint


@router.get('')
def get_study():
    manifest, results, checkpoint = checked_study()
    rows = []
    for profile in results['profiles']:
        for policy in POLICIES:
            runs = [r for r in results['runs'] if r['profile'] == profile and r['policy'] == policy]
            metrics = {}
            for name in ('total_cost', 'p95_latency_ms', 'slo_attainment_lower_bound'):
                metrics[name] = summarize_metric([r['metrics'][name] for r in runs])
            rows.append(dict(profile=profile, policy=policy, **metrics,
                             slo_met_runs=sum(r['metrics']['slo_attainment_lower_bound']
                                              >= manifest['config']['slo_target'] for r in runs),
                             run_count=len(runs)))
    try:
        with (STUDY_ROOT / 'training.csv').open(encoding='utf-8') as handle:
            history = list(csv.DictReader(handle))
        with (STUDY_ROOT / 'validation.csv').open(encoding='utf-8') as handle:
            validation = list(csv.DictReader(handle))
        curve = {}
        returns = [float(r['episode_return']) for r in history]
        for i, row in enumerate(history):
            if i % 10 == 0 or i == len(history) - 1:
                step = int(row['timesteps'])
                recent = returns[max(0, i - 19):i + 1]
                curve[step] = dict(timesteps=step, training_return=sum(recent) / len(recent))
        for row in validation:
            step = int(row['timesteps'])
            curve.setdefault(step, dict(timesteps=step))['validation_return'] = float(row['mean_validation_return'])
    except (OSError, KeyError, ValueError):
        raise HTTPException(503, 'Recorded learning curves could not be read.')
    return dict(checkpoint=dict(name=checkpoint.name, sha256=manifest['checkpoint_sha256'],
                                verified=True, selected_step=manifest['selected_step']),
                manifest=manifest, profiles=results['profiles'], seeds=results['seeds'],
                summaries=rows, runs=results['runs'],
                learning=[curve[step] for step in sorted(curve)],
                limitations=results['limitations'])


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    profile: Profile = 'bursty'
    seed: Literal[1001, 1002, 1003, 1004, 1005] = 1001


@router.post('/replay')
def replay(request: ReplayRequest):
    if not replay_lock.acquire(blocking=False):
        raise HTTPException(429, 'A study replay is already running. Try again shortly.')
    try:
        manifest, results, checkpoint = checked_study()
        if manifest['config'] != CONFIG:
            raise HTTPException(409, 'The current replay protocol differs from the recorded study.')
        records = read_json(f'test_traces/{request.profile}_{request.seed}.json')
        requests = [RequestSchema.model_validate(row) for row in records]
        digest = trace_digest(requests)
        expected = [r for r in results['runs']
                    if r['profile'] == request.profile and r['seed'] == request.seed]
        if not expected or any(r['trace_hash'] != digest for r in expected):
            raise HTTPException(409, 'The test trace differs from the recorded shared workload.')
        fresh = []
        for name in ('fixed', 'ppo'):
            policy = get_policy(name, dict(model_path=str(checkpoint), device='cpu',
                                           fixed_capacity=4, cooldown=CONFIG['cooldown']))
            if name == 'ppo' and policy.model is None:
                raise HTTPException(409, 'The trained model could not be loaded; no fallback is used.')
            metrics, timeline = run_policy(requests, policy, request.seed)
            recorded = next(row['metrics'] for row in expected if row['policy'] == name)
            fresh.append(dict(policy=name, metrics=metrics, timeline=timeline,
                              matches_recorded=metrics == recorded))
        return dict(profile=request.profile, seed=request.seed, request_count=len(requests),
                    trace_hash=digest, checkpoint_sha256=manifest['checkpoint_sha256'],
                    config=CONFIG, source='fresh_checkpoint_inference', runs=fresh)
    finally:
        replay_lock.release()


@router.get('/export/{artifact}')
def export(artifact: Literal['runs', 'manifest', 'checkpoint', 'report']):
    checked_study()
    files = dict(runs=STUDY_ROOT / 'test_runs.csv',
                 manifest=STUDY_ROOT / 'training_manifest.json',
                 checkpoint=STUDY_ROOT / 'ppo_autoscaler.zip',
                 report=ROOT / 'deliverables' / 'InferScale_Report' / 'InferScale_Project_Report.pdf')
    path = files[artifact]
    if not path.is_file():
        raise HTTPException(404, 'This study artifact is unavailable.')
    media = dict(runs='text/csv', manifest='application/json',
                 checkpoint='application/zip', report='application/pdf')
    return FileResponse(path, filename=path.name, media_type=media[artifact])
