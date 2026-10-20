"""Validated benchmark lifecycle and contained report downloads."""

from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from inferscale.config import settings
from inferscale.experiments.persistence import db
from inferscale.experiments.reporting import ARTIFACTS
from inferscale.experiments.statistics import aggregate_results, paired_comparisons
from inferscale.schemas.benchmark import BenchmarkConfig, BenchmarkResult
from inferscale.services.benchmark import BenchmarkService

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])
service = BenchmarkService(db, Path(settings.reports_dir) / "benchmarks")


class BenchmarkRequest(BenchmarkConfig):
    """HTTP clients select model IDs, never filesystem paths or arbitrary datasets."""

    checkpoint_path: None = None
    trace_path: None = None


class BenchmarkJobResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    config: dict
    state: Literal["queued", "running", "completed", "failed"]
    progress: float
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class BenchmarkResultsResponse(BaseModel):
    id: str
    state: str
    result: BenchmarkResult
    aggregates: list[dict]
    paired: list[dict]
    successful_runs: int
    failed_runs: int


def get_record(identifier: UUID):
    record = service.db.get_benchmark(str(identifier))
    if record is None:
        raise HTTPException(404, "Benchmark not found")
    return record


@router.post("", status_code=202, response_model=BenchmarkJobResponse)
def create_benchmark(request: BenchmarkRequest):
    """Queue a benchmark; CPU execution happens in the existing style of background thread."""
    if request.experiment.dataset_id != "synthetic":
        raise HTTPException(
            422,
            "HTTP benchmarks currently support synthetic datasets; "
            "use the local CLI for canonical CSV/JSON traces",
        )
    config = BenchmarkConfig.model_validate(request.model_dump())
    if "ppo" in config.policies:
        model = service.db.get_model(config.model_id) if config.model_id else None
        if not model:
            raise HTTPException(422, "Select a registered model_id with an existing PPO checkpoint")
        root = Path(settings.models_dir).resolve()
        checkpoint = Path(model["path"]).resolve()
        if (
            not checkpoint.is_relative_to(root)
            or checkpoint.suffix != ".zip"
            or not checkpoint.is_file()
        ):
            raise HTTPException(
                422, "Registered PPO checkpoint is missing or outside the model directory"
            )
        config.checkpoint_path = str(checkpoint)
    try:
        identifier = service.submit(config)
    except ValueError as exc:
        raise HTTPException(429, str(exc)) from exc
    return service.db.get_benchmark(identifier)


@router.get("", response_model=list[BenchmarkJobResponse])
def list_benchmarks(limit: int = Query(50, ge=1, le=100)):
    return service.db.list_benchmarks(limit)


@router.get("/{identifier}", response_model=BenchmarkJobResponse)
def get_benchmark(identifier: UUID):
    return get_record(identifier)


@router.get("/{identifier}/results", response_model=BenchmarkResultsResponse)
def get_results(identifier: UUID):
    record = get_record(identifier)
    if record["results"] is None:
        raise HTTPException(409, "No runs are available yet")
    result = BenchmarkResult.model_validate(record["results"])
    successful = sum(run.status == "completed" for run in result.runs)
    return dict(
        id=str(identifier),
        state=record["state"],
        result=result,
        aggregates=aggregate_results(result),
        paired=paired_comparisons(result),
        successful_runs=successful,
        failed_runs=len(result.runs) - successful,
    )


@router.get("/{identifier}/export")
def export_benchmark(
    identifier: UUID, artifact: Literal["runs", "aggregates", "report", "manifest"] = "report"
):
    record = get_record(identifier)
    if record["state"] not in ("completed", "failed"):
        raise HTTPException(409, "Reports are available after the benchmark finishes")
    root = service.reports_root.resolve()
    path = (root / str(identifier) / ARTIFACTS[artifact]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Report artifact not found")
    media = (
        "text/csv"
        if path.suffix == ".csv"
        else ("application/json" if path.suffix == ".json" else "text/markdown")
    )
    return FileResponse(path, media_type=media, filename=f"benchmark-{identifier}-{path.name}")
