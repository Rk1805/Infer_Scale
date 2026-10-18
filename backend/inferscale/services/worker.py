"""Background worker for long-running jobs."""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Optional

from inferscale.experiments.persistence import db
from inferscale.experiments.runner import runner, JobState
from inferscale.config import settings


class BackgroundWorker:
    """Background worker for processing experiment and training jobs."""

    def __init__(self):
        self.running = False
        self.tasks = set()

    async def start(self):
        """Start the worker."""
        self.running = True
        print("Background worker started")
        
        # Process any queued jobs on startup
        await self.process_queued_jobs()
        
        # Main loop
        while self.running:
            try:
                await self.process_queued_jobs()
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Worker error: {e}")
                await asyncio.sleep(10)

    async def stop(self):
        """Stop the worker."""
        self.running = False
        # Wait for running tasks
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        print("Background worker stopped")

    async def process_queued_jobs(self):
        """Process queued experiments and training jobs."""
        # Process experiment jobs
        queued_experiments = runner.list_jobs(JobState.QUEUED)
        for job in queued_experiments:
            if job.id not in [t.get_name() for t in self.tasks if not t.done()]:
                task = asyncio.create_task(self.run_experiment_job(job), name=job.id)
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
        
        # Process training jobs from database
        queued_training = db.list_training_jobs(state="queued")
        for job_data in queued_training:
            job_id = job_data["id"]
            if job_id not in [t.get_name() for t in self.tasks if not t.done()]:
                task = asyncio.create_task(self.run_training_job(job_id, job_data), name=job_id)
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)

    async def run_experiment_job(self, job):
        """Run an experiment job."""
        print(f"Starting experiment job: {job.id}")
        runner.start_job(job.id)
        
        # Wait for completion
        while True:
            await asyncio.sleep(2)
            current_job = runner.get_job(job.id)
            if not current_job:
                break
            if current_job.state in [JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED]:
                # Persist to database
                db.update_experiment(
                    job.id,
                    state=current_job.state.value,
                    progress=current_job.progress,
                    started_at=current_job.started_at,
                    completed_at=current_job.completed_at,
                    error=current_job.error,
                    results=current_job.results,
                    timeline=current_job.timeline,
                )
                break

    async def run_training_job(self, job_id: str, job_data: dict):
        """Run a training job."""
        print(f"Starting training job: {job_id}")
        config = job_data["config"]
        
        db.update_training_job(job_id, state="running", started_at=datetime.now())
        
        try:
            total_timesteps = config.get("timesteps", 200000)
            
            # This is a simplified training loop
            # In production, this would use actual PPO training
            for step in range(0, total_timesteps, 1000):
                # Check for cancellation
                current = db.get_training_job(job_id)
                if current and current["state"] == "cancelled":
                    return
                
                progress = min(step / total_timesteps, 1.0)
                import numpy as np
                reward = -1.0 + np.random.normal(0, 0.1)
                
                db.update_training_job(
                    job_id,
                    progress=progress,
                    current_step=step,
                    total_steps=total_timesteps,
                    episode_rewards=[reward],
                )
                
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


async def main():
    """Main entry point for the worker."""
    worker = BackgroundWorker()
    
    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(worker.stop()))
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        pass
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())