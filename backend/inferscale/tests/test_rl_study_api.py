"""Recorded evidence, artifact integrity, and real dashboard replay integration."""

import hashlib
import shutil

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inferscale.api import rl_study


@pytest.fixture
def study_client(tmp_path, monkeypatch):
    copied = tmp_path / 'study'
    shutil.copytree(rl_study.STUDY_ROOT, copied)
    monkeypatch.setattr(rl_study, 'STUDY_ROOT', copied)
    app = FastAPI()
    app.include_router(rl_study.router)
    return TestClient(app), copied


def test_saved_evidence_has_real_curves_and_all_comparisons(study_client):
    client, _ = study_client
    response = client.get('/api/rl-study')
    assert response.status_code == 200
    data = response.json()
    assert data['checkpoint']['verified']
    assert data['checkpoint']['selected_step'] == 32768
    assert len(data['runs']) == 60
    assert len(data['summaries']) == 12
    assert any(row.get('training_return') is not None for row in data['learning'])
    assert any(row.get('validation_return') is not None for row in data['learning'])
    ppo = [row for row in data['summaries'] if row['policy'] == 'ppo']
    assert sum(row['slo_met_runs'] for row in ppo) == 15


def test_actual_dashboard_replay_matches_frozen_evaluation(study_client):
    client, _ = study_client
    response = client.post('/api/rl-study/replay', json={'profile': 'bursty', 'seed': 1001})
    assert response.status_code == 200
    data = response.json()
    assert data['source'] == 'fresh_checkpoint_inference'
    assert data['request_count'] > 1000
    assert {row['policy'] for row in data['runs']} == {'fixed', 'ppo'}
    assert all(row['matches_recorded'] for row in data['runs'])
    ppo = next(row for row in data['runs'] if row['policy'] == 'ppo')
    fixed = next(row for row in data['runs'] if row['policy'] == 'fixed')
    assert ppo['metrics']['slo_attainment_lower_bound'] == 1
    assert fixed['metrics']['slo_attainment_lower_bound'] < .95
    assert len(ppo['timeline']) == 36


def test_corrupt_checkpoint_is_not_presented_as_verified(study_client):
    client, copied = study_client
    (copied / 'ppo_autoscaler.zip').write_bytes(b'corrupt')
    assert client.get('/api/rl-study').status_code == 409
    assert client.post('/api/rl-study/replay', json={}).status_code == 409


def test_modified_test_trace_cannot_be_replayed(study_client):
    client, copied = study_client
    (copied / 'test_traces' / 'bursty_1001.json').write_text('[]')
    assert client.post('/api/rl-study/replay', json={}).status_code == 409


def test_missing_artifacts_give_actionable_error(study_client):
    client, copied = study_client
    (copied / 'training_manifest.json').unlink()
    response = client.get('/api/rl-study')
    assert response.status_code == 404
    assert 'README' in response.json()['detail']


@pytest.mark.parametrize('payload', [
    {'profile': '../../private'}, {'seed': 42}, {'checkpoint_path': 'other.zip'},
])
def test_replay_only_accepts_bounded_study_inputs(study_client, payload):
    client, _ = study_client
    assert client.post('/api/rl-study/replay', json=payload).status_code == 422


def test_concurrent_replay_is_bounded(study_client):
    client, _ = study_client
    rl_study.replay_lock.acquire()
    try:
        assert client.post('/api/rl-study/replay', json={}).status_code == 429
    finally:
        rl_study.replay_lock.release()


def test_downloads_expose_only_named_evidence(study_client):
    client, copied = study_client
    response = client.get('/api/rl-study/export/checkpoint')
    assert response.status_code == 200
    assert hashlib.sha256(response.content).hexdigest() == hashlib.sha256(
        (copied / 'ppo_autoscaler.zip').read_bytes()).hexdigest()
    assert client.get('/api/rl-study/export/runs').headers['content-type'].startswith('text/csv')
    assert client.get('/api/rl-study/export/unknown').status_code == 422
