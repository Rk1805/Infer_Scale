"""Bounded, thread-based benchmark jobs using the existing SQLite and job conventions."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from inferscale.experiments.benchmark import run_benchmark
from inferscale.experiments.persistence import ExperimentDB
from inferscale.experiments.reporting import write_reports
from inferscale.experiments.runner import JobState
from inferscale.schemas.benchmark import BenchmarkConfig


class BenchmarkService:
    """Run CPU simulations off the event loop, persisting progress and partial results.

    Like the existing experiment runner this is a single-process service. Use one
    API worker. One benchmark executes at a time; at most eight may be outstanding.
    """

    def __init__(self, database: ExperimentDB, reports_root: str | Path):
        self.db = database
        self.reports_root = Path(reports_root)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inferscale-benchmark")
        self._lock = threading.Lock()
        self._active = 0

    def submit(self, config: BenchmarkConfig) -> str:
        """Persist a validated job before submitting it to the bounded executor."""
        with self._lock:
            if self._active >= 8:
                raise ValueError("Benchmark queue is full; retry after a job finishes")
            self._active += 1
        identifier = str(uuid.uuid4())
        try:
            self.db.create_benchmark(identifier, config.model_dump(mode="json"))
            self.executor.submit(self._execute, identifier, config)
        except Exception:
            with self._lock:
                self._active -= 1
            raise
        return identifier

    def _execute(self, identifier: str, config: BenchmarkConfig) -> None:
        runs = []

        def progress(run, completed, total):
            runs.append(run.model_dump(mode="json"))
            self.db.update_benchmark(
                identifier,
                progress=completed / total,
                results={"config": config.model_dump(mode="json"), "runs": runs},
            )

        try:
            self.db.update_benchmark(
                identifier,
                state=JobState.RUNNING.value,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            result = run_benchmark(config, on_run=progress)
            write_reports(result, self.reports_root, identifier)
            succeeded = any(run.status == "completed" for run in result.runs)
            self.db.update_benchmark(
                identifier,
                state=JobState.COMPLETED.value if succeeded else JobState.FAILED.value,
                progress=1,
                results=result.model_dump(mode="json"),
                error=None if succeeded else "All benchmark runs failed; inspect per-run errors",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            self.db.update_benchmark(
                identifier,
                state=JobState.FAILED.value,
                error=f"{type(exc).__name__}: {exc}",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        finally:
            with self._lock:
                self._active -= 1

    def close(self) -> None:
        """Finish outstanding work on orderly shutdown."""
        self.executor.shutdown(wait=True)
