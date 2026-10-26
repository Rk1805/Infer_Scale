"""FastAPI application for InferScale."""

from contextlib import asynccontextmanager
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from inferscale.config import settings
from inferscale.experiments.runner import runner, ExperimentJob, JobState
from inferscale.experiments.persistence import db
from inferscale.schemas.experiment import ExperimentConfig, ExperimentStatus, PolicyType
from inferscale.simulator.engine import run_simulation
from inferscale.data.synth import SYNTHETIC_WORKLOADS
from inferscale.policies import get_policy
from inferscale.api.benchmarks import router as benchmark_router
from inferscale.api.rl_study import router as rl_study_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting InferScale API...")
    from inferscale.api.benchmarks import service
    service.db.interrupt_benchmarks()
    yield
    # Shutdown
    print("Shutting down InferScale API...")
    import asyncio
    await asyncio.to_thread(service.close)


app = FastAPI(
    title="InferScale API",
    description="Autonomous LLM Inference Autoscaling with Reinforcement Learning",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(benchmark_router)
app.include_router(rl_study_router)


# Health endpoint
@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": "2026-10-18T00:00:00Z"}


# Dataset endpoints
@app.get("/api/datasets")
async def list_datasets():
    datasets = db.list_datasets()
    # Add synthetic dataset
    datasets.insert(0, {
        "id": "synthetic",
        "name": "Synthetic Workloads",
        "source": "generated",
        "record_count": 0,
        "duration_seconds": 0,
        "hash": "synthetic",
        "created_at": "2026-01-01T00:00:00Z",
    })
    return datasets


@app.post("/api/datasets/validate")
async def validate_dataset(file: bytes = None):
    """Validate a dataset file (placeholder)."""
    return {
        "valid": True,
        "record_count": 1000,
        "errors": [],
        "warnings": [],
        "preview": [],
    }


@app.get("/api/datasets/{dataset_id}/stats")
async def get_dataset_stats(dataset_id: str):
    if dataset_id == "synthetic":
        return {
            "id": "synthetic",
            "record_count": 10000,
            "duration_seconds": 3600,
            "prompt_tokens": {"min": 10, "max": 500, "mean": 100, "median": 80, "std": 50, "p50": 80, "p95": 200, "p99": 400},
            "output_tokens": {"min": 5, "max": 300, "mean": 50, "median": 40, "std": 30, "p50": 40, "p95": 150, "p99": 250},
            "arrival_rate": [{"timestamp": i * 60, "value": 10 + (i % 5) * 2} for i in range(60)],
            "burst_detection": [],
            "missing_values": 0,
            "invalid_rows": 0,
            "schema": {"id": "string", "arrival_time_seconds": "float", "prompt_tokens": "int", "output_tokens": "int"},
        }
    
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return dataset


# Experiment endpoints
@app.post("/api/experiments")
async def create_experiment(config: ExperimentConfig, background_tasks: BackgroundTasks):
    """Create and start a new experiment."""
    job = runner.create_job(config)
    runner.start_job(job.id)
    
    # Persist to database
    db.create_experiment(job.id, config.model_dump(), "queued")
    
    return {
        "id": job.id,
        "config": config.model_dump(),
        "state": job.state.value,
        "progress": job.progress,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
        "results": job.results,
        "timeline": job.timeline,
    }


@app.get("/api/experiments")
async def list_experiments(status: Optional[str] = None):
    """List all experiments."""
    job_state = JobState(status) if status else None
    jobs = runner.list_jobs(job_state)
    
    # Also get from database for persistence across restarts
    db_experiments = db.list_experiments(status)
    
    # Merge (in-memory takes precedence)
    seen = set()
    result = []
    for job in jobs:
        seen.add(job.id)
        result.append({
            "id": job.id,
            "config": job.config.model_dump(),
            "state": job.state.value,
            "progress": job.progress,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error,
            "results": job.results,
            "timeline": job.timeline,
        })
    
    for exp in db_experiments:
        if exp["id"] not in seen:
            result.append(exp)
    
    return result


@app.get("/api/experiments/{exp_id}")
async def get_experiment(exp_id: str):
    """Get a specific experiment."""
    job = runner.get_job(exp_id)
    if job:
        return {
            "id": job.id,
            "config": job.config.model_dump(),
            "state": job.state.value,
            "progress": job.progress,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error,
            "results": job.results,
            "timeline": job.timeline,
        }
    
    # Check database
    exp = db.get_experiment(exp_id)
    if exp:
        return exp
    
    raise HTTPException(status_code=404, detail="Experiment not found")


@app.post("/api/experiments/{exp_id}/cancel")
async def cancel_experiment(exp_id: str):
    """Cancel a running experiment."""
    success = runner.cancel_job(exp_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel experiment")
    
    db.update_experiment(exp_id, state="cancelled")
    return {"success": True}


@app.get("/api/experiments/{exp_id}/metrics")
async def get_experiment_metrics(exp_id: str):
    """Get experiment metrics."""
    job = runner.get_job(exp_id)
    if job and job.results:
        return job.results
    
    exp = db.get_experiment(exp_id)
    if exp and exp.get("results"):
        return exp["results"]
    
    raise HTTPException(status_code=404, detail="Metrics not found")


@app.get("/api/experiments/{exp_id}/timeline")
async def get_experiment_timeline(exp_id: str):
    """Get experiment timeline for visualization."""
    job = runner.get_job(exp_id)
    if job and job.timeline:
        return job.timeline
    
    exp = db.get_experiment(exp_id)
    if exp and exp.get("timeline"):
        return exp["timeline"]
    
    raise HTTPException(status_code=404, detail="Timeline not found")


@app.get("/api/experiments/{exp_id}/export")
async def export_experiment(exp_id: str, format: str = "json"):
    """Export experiment results."""
    job = runner.get_job(exp_id)
    if not job:
        exp = db.get_experiment(exp_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        job = type('obj', (object,), exp)()
    
    if format == "json":
        import json
        content = json.dumps({
            "id": exp_id,
            "config": job.config if hasattr(job, 'config') else job.get("config"),
            "results": job.results if hasattr(job, 'results') else job.get("results"),
            "timeline": job.timeline if hasattr(job, 'timeline') else job.get("timeline"),
        }, indent=2, default=str)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=experiment-{exp_id}.json"},
        )
    elif format == "csv":
        # Generate CSV from results
        results = job.results if hasattr(job, 'results') else job.get("results")
        if not results:
            raise HTTPException(status_code=404, detail="No results to export")
        
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, v])
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=experiment-{exp_id}.csv"},
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid format")


# Training endpoints
@app.post("/api/training")
async def start_training(config: dict, background_tasks: BackgroundTasks):
    """Start a new PPO training job."""
    import uuid
    job_id = str(uuid.uuid4())[:8]
    
    db.create_training_job(job_id, config, "queued")
    
    # Start training in background
    background_tasks.add_task(run_training_background, job_id, config)
    
    return {
        "id": job_id,
        "config": config,
        "state": "queued",
        "progress": 0.0,
        "current_step": 0,
        "total_steps": config.get("timesteps", 200000),
        "episode_rewards": [],
        "eval_rewards": [],
        "started_at": None,
        "completed_at": None,
        "error": None,
        "model_path": None,
    }


@app.get("/api/training/{job_id}")
async def get_training(job_id: str):
    """Get training job status."""
    job = db.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@app.get("/api/models")
async def list_models():
    """List all trained models."""
    return db.list_models()


@app.post("/api/evaluations")
async def start_evaluation(config: dict, background_tasks: BackgroundTasks):
    """Start a model evaluation."""
    import uuid
    job_id = str(uuid.uuid4())[:8]
    
    # This would run evaluation in background
    # For now, return a placeholder
    return {
        "id": job_id,
        "config": config,
        "state": "queued",
        "progress": 0.0,
        "results": None,
        "started_at": None,
        "completed_at": None,
        "error": None,
    }


@app.get("/api/reports/{exp_id}")
async def get_report(exp_id: str):
    """Generate a markdown report for an experiment."""
    job = runner.get_job(exp_id)
    if not job:
        exp = db.get_experiment(exp_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        job = type('obj', (object,), exp)()
    
    # Generate markdown report
    metrics = job.results if hasattr(job, 'results') else job.get("results")
    config = job.config if hasattr(job, 'config') else job.get("config")
    
    if not metrics:
        raise HTTPException(status_code=404, detail="No results to report")
    
    report = generate_markdown_report(exp_id, config, metrics)
    
    return {"id": exp_id, "markdown": report, "created_at": "2026-10-18T00:00:00Z"}


# SSE endpoint for live updates
@app.get("/api/events/experiments/{exp_id}")
async def experiment_events(exp_id: str):
    """SSE endpoint for experiment progress updates."""
    async def event_generator():
        import asyncio
        last_progress = -1
        while True:
            job = runner.get_job(exp_id)
            if not job:
                exp = db.get_experiment(exp_id)
                if exp:
                    yield f"data: {json.dumps(exp)}\n\n"
                    break
            
            if job.progress != last_progress:
                yield f"data: {json.dumps({'progress': job.progress, 'state': job.state.value})}\n\n"
                last_progress = job.progress
            
            if job.state in [JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED]:
                yield f"data: {json.dumps({'progress': 1.0, 'state': job.state.value, 'results': job.results})}\n\n"
                break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def run_training_background(job_id: str, config: dict):
    """Background task for PPO training."""
    from datetime import datetime
    import numpy as np
    
    db.update_training_job(job_id, state="running", started_at=datetime.now())
    
    try:
        total_timesteps = config.get("timesteps", 200000)
        
        # Simulate training progress (replace with actual PPO training)
        for step in range(0, total_timesteps, 1000):
            # Check for cancellation
            job = db.get_training_job(job_id)
            if job and job["state"] == "cancelled":
                return
            
            progress = min(step / total_timesteps, 1.0)
            reward = -1.0 + np.random.normal(0, 0.1)  # Simulated reward
            
            db.update_training_job(
                job_id,
                progress=progress,
                current_step=step,
                total_steps=total_timesteps,
                episode_rewards=[reward],
            )
            
            # Small delay to simulate work
            import asyncio
            await asyncio.sleep(0.1)
        
        # Training complete
        model_path = f"{settings.models_dir}/ppo_{job_id}.zip"
        db.update_training_job(
            job_id,
            state="completed",
            progress=1.0,
            current_step=total_timesteps,
            completed_at=datetime.now(),
            model_path=model_path,
        )
        
        # Save model metadata
        db.create_model(
            f"model_{job_id}",
            f"PPO-{job_id}",
            model_path,
            config,
            {"final_reward": -0.5, "avg_eval_reward": -0.6, "training_time_seconds": 30},
        )
        
    except Exception as e:
        db.update_training_job(
            job_id,
            state="failed",
            error=str(e),
            completed_at=datetime.now(),
        )


def generate_markdown_report(exp_id: str, config: dict, metrics: dict) -> str:
    """Generate markdown report for an experiment."""
    md = f"# Experiment Report: {exp_id}\n\n"
    md += f"**Generated:** 2026-10-18T00:00:00Z\n\n"
    
    md += "## Configuration\n\n"
    md += "| Parameter | Value |\n"
    md += "|-----------|-------|\n"
    for k, v in config.items():
        md += f"| {k} | {v} |\n"
    md += "\n"
    
    md += "## Results\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    for k, v in metrics.items():
        md += f"| {k} | {v} |\n"
    md += "\n"
    
    md += "---\n*Report generated by InferScale*"
    return md
