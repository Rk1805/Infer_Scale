"""SQLite persistence for experiments, models, and training jobs."""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from inferscale.config import settings


class ExperimentDB:
    """SQLite database for experiment persistence."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.db_url.replace("sqlite:///", "")
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    state TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    results TEXT,
                    timeline TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    config TEXT NOT NULL,
                    metrics TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS training_jobs (
                    id TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    state TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    current_step INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 0,
                    episode_rewards TEXT,
                    eval_rewards TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    model_path TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    hash TEXT NOT NULL,
                    schema TEXT,
                    stats TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS benchmarks (
                    id TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'queued',
                    progress REAL NOT NULL DEFAULT 0,
                    results TEXT,
                    error TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_experiments_state ON experiments(state);
                CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
                CREATE INDEX IF NOT EXISTS idx_models_created ON models(created_at);
                CREATE INDEX IF NOT EXISTS idx_training_jobs_state ON training_jobs(state);
            """)

    @contextmanager
    def _get_conn(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Benchmark records are additive; legacy experiment/model data is untouched.
    def create_benchmark(self, benchmark_id: str, config: Dict) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute(
                "INSERT INTO benchmarks (id, config, created_at) VALUES (?, ?, ?)",
                (benchmark_id, json.dumps(config), datetime.now().isoformat()),
            )

    def update_benchmark(self, benchmark_id: str, **values) -> None:
        allowed = {"state", "progress", "results", "error", "started_at", "completed_at"}
        if not values or not set(values) <= allowed:
            raise ValueError("Invalid benchmark update")
        fields = list(values)
        params = [json.dumps(values[key], allow_nan=False) if key == "results" else values[key]
                  for key in fields]
        with self._lock, self._get_conn() as conn:
            conn.execute(
                "UPDATE benchmarks SET " + ", ".join(f"{key} = ?" for key in fields) + " WHERE id = ?",
                [*params, benchmark_id],
            )

    def get_benchmark(self, benchmark_id: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM benchmarks WHERE id = ?", (benchmark_id,)).fetchone()
            return self._deserialize_benchmark(row) if row else None

    def list_benchmarks(self, limit: int = 100) -> List[Dict]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM benchmarks ORDER BY created_at DESC LIMIT ?",
                                (limit,)).fetchall()
            return [self._deserialize_benchmark(row) for row in rows]

    def interrupt_benchmarks(self) -> None:
        """A process restart cannot resume an in-memory simulation safely."""
        with self._lock, self._get_conn() as conn:
            conn.execute(
                "UPDATE benchmarks SET state='failed', error=?, completed_at=? "
                "WHERE state IN ('queued', 'running')",
                ("Server stopped before benchmark completion; submit a new benchmark to retry",
                 datetime.now().isoformat()),
            )

    @staticmethod
    def _deserialize_benchmark(row) -> Dict:
        data = dict(row)
        data["config"] = json.loads(data["config"])
        data["results"] = json.loads(data["results"]) if data["results"] else None
        return data

    # Experiment methods
    def create_experiment(self, exp_id: str, config: Dict, state: str = "queued") -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO experiments (id, config, state, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (exp_id, json.dumps(config), state, datetime.now().isoformat()),
                )

    def update_experiment(
        self,
        exp_id: str,
        state: Optional[str] = None,
        progress: Optional[float] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error: Optional[str] = None,
        results: Optional[Dict] = None,
        timeline: Optional[List] = None,
    ) -> None:
        with self._lock:
            with self._get_conn() as conn:
                updates = []
                params = []
                if state is not None:
                    updates.append("state = ?")
                    params.append(state)
                if progress is not None:
                    updates.append("progress = ?")
                    params.append(progress)
                if started_at is not None:
                    updates.append("started_at = ?")
                    params.append(started_at.isoformat())
                if completed_at is not None:
                    updates.append("completed_at = ?")
                    params.append(completed_at.isoformat())
                if error is not None:
                    updates.append("error = ?")
                    params.append(error)
                if results is not None:
                    updates.append("results = ?")
                    params.append(json.dumps(results))
                if timeline is not None:
                    updates.append("timeline = ?")
                    params.append(json.dumps(timeline))
                
                if updates:
                    params.append(exp_id)
                    conn.execute(
                        f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?",
                        params,
                    )

    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
            if row:
                return self._deserialize_experiment(row)
            return None

    def list_experiments(self, state: Optional[str] = None, limit: int = 100) -> List[Dict]:
        with self._get_conn() as conn:
            if state:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE state = ? ORDER BY created_at DESC LIMIT ?",
                    (state, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._deserialize_experiment(row) for row in rows]

    def delete_experiment(self, exp_id: str) -> bool:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
                return cursor.rowcount > 0

    # Model methods
    def create_model(self, model_id: str, name: str, path: str, config: Dict, metrics: Optional[Dict] = None) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO models (id, name, path, config, metrics)
                       VALUES (?, ?, ?, ?, ?)""",
                    (model_id, name, path, json.dumps(config), json.dumps(metrics) if metrics else None),
                )

    def get_model(self, model_id: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def list_models(self, limit: int = 100) -> List[Dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM models ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def update_model_metrics(self, model_id: str, metrics: Dict) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE models SET metrics = ? WHERE id = ?",
                    (json.dumps(metrics), model_id),
                )

    # Training job methods
    def create_training_job(self, job_id: str, config: Dict, state: str = "queued") -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO training_jobs (id, config, state, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (job_id, json.dumps(config), state, datetime.now().isoformat()),
                )

    def update_training_job(
        self,
        job_id: str,
        state: Optional[str] = None,
        progress: Optional[float] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        episode_rewards: Optional[List] = None,
        eval_rewards: Optional[List] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        with self._lock:
            with self._get_conn() as conn:
                updates = []
                params = []
                if state is not None:
                    updates.append("state = ?")
                    params.append(state)
                if progress is not None:
                    updates.append("progress = ?")
                    params.append(progress)
                if current_step is not None:
                    updates.append("current_step = ?")
                    params.append(current_step)
                if total_steps is not None:
                    updates.append("total_steps = ?")
                    params.append(total_steps)
                if episode_rewards is not None:
                    updates.append("episode_rewards = ?")
                    params.append(json.dumps(episode_rewards))
                if eval_rewards is not None:
                    updates.append("eval_rewards = ?")
                    params.append(json.dumps(eval_rewards))
                if started_at is not None:
                    updates.append("started_at = ?")
                    params.append(started_at.isoformat())
                if completed_at is not None:
                    updates.append("completed_at = ?")
                    params.append(completed_at.isoformat())
                if error is not None:
                    updates.append("error = ?")
                    params.append(error)
                if model_path is not None:
                    updates.append("model_path = ?")
                    params.append(model_path)
                
                if updates:
                    params.append(job_id)
                    conn.execute(
                        f"UPDATE training_jobs SET {', '.join(updates)} WHERE id = ?",
                        params,
                    )

    def get_training_job(self, job_id: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def list_training_jobs(self, state: Optional[str] = None, limit: int = 100) -> List[Dict]:
        with self._get_conn() as conn:
            if state:
                rows = conn.execute(
                    "SELECT * FROM training_jobs WHERE state = ? ORDER BY created_at DESC LIMIT ?",
                    (state, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]

    # Dataset methods
    def create_dataset(self, dataset_id: str, name: str, source: str, record_count: int, 
                       duration_seconds: float, hash: str, schema: Optional[Dict] = None,
                       stats: Optional[Dict] = None) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO datasets (id, name, source, record_count, duration_seconds, hash, schema, stats)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (dataset_id, name, source, record_count, duration_seconds, hash,
                     json.dumps(schema) if schema else None, json.dumps(stats) if stats else None),
                )

    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def list_datasets(self) -> List[Dict]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def _deserialize_experiment(self, row: sqlite3.Row) -> Dict:
        """Deserialize experiment row from database."""
        return {
            "id": row["id"],
            "config": json.loads(row["config"]),
            "state": row["state"],
            "progress": row["progress"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "error": row["error"],
            "results": json.loads(row["results"]) if row["results"] else None,
            "timeline": json.loads(row["timeline"]) if row["timeline"] else [],
            "created_at": row["created_at"],
        }


# Global database instance
db = ExperimentDB()
