# InferScale

**Autonomous LLM Inference Autoscaling with Reinforcement Learning**

**Current scope:** a CPU-based simulation and research dashboard. PPO training is
available through the CLI; live GPU utilisation, inference batch-size actions,
verified Azure/BurstGPT training, and Kubernetes HPA validation are not yet complete.
See the [implementation audit against the project description](docs/project_description_audit.md)
for supported behavior, source evidence, validation limits, and remaining work.

A trained synthetic PPO checkpoint, 60 held-out policy comparisons, and the
two-page assignment report are now included. See [the reproducible study](docs/rl_assignment_study.md)
and [trained-agent artifacts](artifacts/rl_study/README.md) for measured results
and their limits; PPO improves deadline reliability here at higher resource cost.

InferScale investigates whether reinforcement learning can reduce the compute cost of LLM inference serving while maintaining response-time service-level objectives (SLOs). The system compares four autoscaling policies:

1. **Fixed Capacity** - Moves toward a configured replica target under scaling constraints
2. **Threshold Autoscaling** - Scales based on utilization/queue pressure thresholds
3. **Queue-Based Autoscaling** - Reacts to estimated waiting work per replica
4. **PPO (RL)** - Trains a scaling policy via Proximal Policy Optimization

## Architecture

```
InferScale/
├── backend/                 # Python backend (FastAPI + RL + Simulator)
│   └── inferscale/
│       ├── api/            # FastAPI endpoints
│       ├── cli.py          # Command-line interface
│       ├── config.py       # Configuration management
│       ├── data/           # Data loading, synthetic workloads, adapters
│       ├── experiments/    # Experiment orchestration & persistence
│       ├── policies/       # Baseline policies + PPO policy
│       ├── rl/             # Gymnasium environment + reward function
│       ├── schemas/        # Pydantic schemas
│       ├── simulator/      # Discrete-event simulation engine
│       ├── services/       # Background worker
│       └── tests/          # Unit & integration tests
├── frontend/               # React + TypeScript + Vite dashboard
│   └── src/
│       ├── components/     # Reusable UI components
│       ├── hooks/          # React hooks for data fetching
│       ├── lib/            # API client & utilities
│       └── pages/          # Dashboard pages
├── docker-compose.yml      # Local deployment
├── Dockerfile.api          # API container
├── Dockerfile.frontend     # Frontend container
├── Makefile               # Development commands
└── pyproject.toml         # Python dependencies
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker & Docker Compose (optional)

### Installation

```bash
# Clone and enter directory
git clone https://github.com/Rk1805/Infer_Scale.git
cd Infer_Scale

# Create a Python environment
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then run:

```bash
python -m pip install -e ".[dev,test]"
npm --prefix frontend install

# Optional: generate a synthetic trace
python -m inferscale.cli generate-data --output data/synthetic.csv
```

### Development

```bash
# Start both API and frontend (two terminals)
python -m uvicorn inferscale.api.main:app --host 127.0.0.1 --port 8000  # Terminal 1
npm --prefix frontend run dev                                      # Terminal 2

# Or use Docker
make docker-up
```

### Training & Evaluation

```bash
# Evaluate the bundled assignment checkpoint on reserved synthetic traces
python -m inferscale.experiments.rl_study evaluate

# Retrain without overwriting the bundled evidence
python -m inferscale.experiments.rl_study train --directory artifacts/reproduction
python -m inferscale.experiments.rl_study evaluate --directory artifacts/reproduction

# Original CLI training (a separate synthetic-workload setup)
python -m inferscale.cli train --timesteps 200000

# Reward-only evaluation of that original CLI model
python -m inferscale.cli evaluate --model-path models/ppo_full.zip --episodes 10

# Run the original policy comparison benchmark
python -m inferscale.cli benchmark --policies fixed threshold queue
```

## Assignment Agent and Reproducible Study

### Show it on the dashboard

With API and frontend running, open **RL Study** in the sidebar or
<http://127.0.0.1:5173/rl-study>. This page reads the actual saved study artifacts:

1. Show the verified checkpoint, training/validation curves, and held-out totals.
2. Switch between bursty, steady, and shifted traffic to compare cost, on-time
   attainment, and p95 latency with 95% run-level intervals.
3. Choose a reserved request seed and click **Run PPO replay**. The API loads the
   trained checkpoint and runs fresh inference alongside fixed capacity on the
   exact same trace, then displays whether the metrics match the saved study.
4. Play, pause, or scrub through decisions to explain active/starting replicas,
   queue length, and requested actions. Playback reveals completed simulation
   records; it is not real-time GPU telemetry.
5. Inspect individual runs, split/configuration details, or download the report,
   checkpoint, configuration, and raw results.

Replay is a bounded 180-second CPU simulation, not retraining. Corrupt/missing
checkpoints and modified traces are rejected; inference does not substitute a
baseline. The older Training Lab progress flow is explicitly labelled as a
placeholder and links to this real study.

### Train and reproduce

The assignment study uses the existing simulator, policies, and interval reward
with a separate seeded protocol in
[`rl_study.py`](backend/inferscale/experiments/rl_study.py).
It trains PPO for 32,768 steps on independently generated synthetic traces,
selects a **trained** checkpoint by validation return, and evaluates that frozen
checkpoint on reserved test seeds. Training uses bursty and steady profiles;
the shifted profile is reserved for testing. Arrivals, token counts, and phase
rates vary by trace seed. These are synthetic CPU simulations, not Azure/BurstGPT
or measured GPU results.

Artifacts are written under [`artifacts/rl_study/`](artifacts/rl_study/):

- `ppo_autoscaler.zip`: selected trained agent, including fixed input scaling.
- Retraining also generates `ppo_final.zip`; the release versions the selected
  `ppo_autoscaler.zip` rather than duplicate checkpoints.
- `training_manifest.json`: configuration, versions, seed partitions, selected
  step, checkpoint hash, and code provenance.
- `training.csv` and `validation.csv`: measured learning/selection histories.
- `test_evaluation.json` and `test_runs.csv`: all policy results on shared test traces.
- `test_traces/` and representative timelines: exact requests and scaling behaviour.

Install the source dependencies before loading the trusted checkpoint: its
feature extractor is `inferscale.rl.features.ScaledObservations`. No external
normalisation file is needed. For example:

```python
from stable_baselines3 import PPO
from inferscale.experiments.rl_study import CONFIG, REWARD, workload
from inferscale.rl.env import InferScaleEnv

model = PPO.load("artifacts/rl_study/ppo_autoscaler.zip", device="cpu")
env = InferScaleEnv(workload("bursty", 1001), config=CONFIG, reward_config=REWARD)
observation, _ = env.reset()
action, _ = model.predict(observation, deterministic=True)
```

To preserve existing evidence when retraining, choose a fresh directory:

```bash
python -m inferscale.experiments.rl_study train --directory artifacts/reproduction
python -m inferscale.experiments.rl_study evaluate --directory artifacts/reproduction
```

The report chart is regenerated with
`python deliverables/InferScale_Report/generate_assignment_plot.py`.
The two-page PDF and LaTeX source are in `deliverables/InferScale_Report/`.
One training seed is used; the test seeds measure workload variation and do not
establish robustness across independently trained agents.
The released study also compares entropy settings 0 and 0.02 on validation;
`selection.json` preserves that choice and the candidate evidence. See
[the complete study protocol](docs/rl_assignment_study.md) for both-candidate
reproduction commands and simulator corrections.

## Advanced Benchmarking Contribution

The Policy Comparison page now includes **Advanced Benchmarking**: shared workload
traces across policies, independent workload seeds, run-level statistical summaries,
95% Student's t intervals, matched policy differences, and downloadable CSV/Markdown
reports with reproducibility metadata. PPO evaluation requires an existing checkpoint
and never falls back to another policy.

```bash
python -m inferscale.cli benchmark --policies fixed threshold queue --seeds 11 22 33 44 55 --workloads flash_crowd --duration 600
```

See [Advanced Benchmarking: methodology and assessment guide](docs/advanced_benchmarking.md)
for architecture, API endpoints, tests, report exports, and interpretation. Results
are analytical simulation estimates; the inherited cost metric accounts for startup
charges only in legacy results. New runs use runtime accounting v2; see
[cost accounting and SLO evaluation](docs/cost_and_slo.md) for billing rules,
deadline handling, matched savings, and dashboard usage.

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | System health, dataset availability, latest experiment/model status, quick actions |
| **Experiment Studio** | Configure and launch experiments with validated form |
| **Simulation** | Live/replay visualization of replicas, queue, latency, timeline |
| **Policy Comparison** | Compare all 4 policies on same workload with charts & tables |
| **Training Lab** | Configure PPO, monitor training curves, manage checkpoints |
| **Dataset Explorer** | Inspect dataset schema, token distributions, arrival rates, bursts |
| **History & Reports** | Persisted experiments, filtering, export, markdown reports |

## Simulator

The discrete-event simulator models:
- Timestamped request arrivals (FIFO queue)
- Multiple replicas with startup delays
- Configurable per-replica service capacity
- Prompt prefill + token decode phases
- In-flight request tracking
- Safe scale-down (drains in-flight requests)
- Cooldowns and pending scaling actions
- SLA deadlines and violation tracking
- Analytical service time model (calibratable)

**No future information leakage**: Policies only observe current/past state.

## RL Environment

Gymnasium-compatible environment with:
- **Observation**: 12 raw features (queue, arrival rate, replicas, latency, SLA rate, cooldown, etc.). The assignment checkpoint stores fixed input scaling inside its policy.
- **Action Space**: Discrete(3) - Scale Down / Hold / Scale Up
- **Reward**: `r = -(α·C + β·V + γ·Q + δ·S)` where:
  - C = normalized compute cost
  - V = SLA violation penalty
  - Q = normalized queue pressure
  - S = scaling churn penalty

## Experiments

Reproducible evaluation protocol:
1. **Training workloads** - Mixed steady/periodic/burst synthetic traces
2. **Validation workloads** - For hyperparameter tuning
3. **Held-out test workloads** - For final evaluation
4. **Multiple seeds** - Variability estimation
5. **Workload shift tests** - Generalization assessment

Metrics computed from actual simulation events (no survivorship bias).

## API Endpoints

```
GET  /api/health
GET  /api/datasets
POST /api/datasets/validate
GET  /api/datasets/{id}/stats
POST /api/experiments
GET  /api/experiments
GET  /api/experiments/{id}
POST /api/experiments/{id}/cancel
GET  /api/experiments/{id}/metrics
GET  /api/experiments/{id}/timeline
GET  /api/experiments/{id}/export
POST /api/training
GET  /api/training/{id}
GET  /api/models
POST /api/evaluations
GET  /api/reports/{id}
GET  /api/events/experiments/{id}  # SSE for live updates
```

## Data Adapters

Built-in adapters for:
- **Azure LLM Inference Dataset 2024** - `azurellm`
- **BurstGPT** - `burstgpt`

```python
from inferscale.data import load_dataset
requests = load_dataset("path/to/trace.csv", "azurellm")
```

Synthetic workloads: `steady`, `periodic`, `flash_crowd`, `mixed`

## Configuration

Environment variables (see `.env.example`):

```bash
INFERSCALE_DB_URL=sqlite:///./experiments.db
INFERSCALE_STARTUP_DELAY=10
INFERSCALE_COOLDOWN=30
INFERSCALE_MIN_REPLICAS=1
INFERSCALE_MAX_REPLICAS=8
INFERSCALE_PPO_TIMESTEPS=200000
INFERSCALE_API_HOST=0.0.0.0
INFERSCALE_API_PORT=8000
```

## Testing

```bash
# Run all tests
make test

# Specific test suites
make test-simulator
make test-policies
make test-rl
make test-api
```

## Docker Deployment

```bash
# Build and start
make docker-up

# View logs
make docker-logs

# Stop
make docker-down
```

Persistent volumes: `data/`, `models/`, `experiments/`, `reports/`

## Limitations

- **Service model**: Analytical (not cycle-accurate GPU simulation). Does not replicate vLLM/TensorRT-LLM exactly.
- **No GPU required**: Runs entirely on CPU for development.
- **Single-machine**: SQLite persistence, threading-based background jobs.
- **Batch size fixed**: Dynamic batching not modeled in initial version.
- **PPO convergence**: Full training requires 200k+ timesteps; smoke test uses 5k.

## Research Methodology

1. **Workload generation** - Synthetic + real traces (Azure, BurstGPT)
2. **Simulator validation** - Deterministic, no future leakage, configurable service model
3. **Baseline tuning** - Thresholds tuned on validation set only
4. **RL training** - PPO on training workloads, evaluated on held-out test set
5. **Metrics** - Cost, latency (p50/p95/p99), SLA violations, goodput, scaling events
6. **Statistical rigor** - Multiple seeds, confidence intervals, Pareto frontiers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `make test` and `make lint`
5. Submit a PR

## License

MIT License - see LICENSE file for details.

## Citation

If you use InferScale in your research, please cite:

```bibtex
@software{inferscale2026,
  title={InferScale: Autonomous LLM Inference Autoscaling with Reinforcement Learning},
  author={InferScale Team},
  year={2026},
  url={https://github.com/your-org/inferscale}
}
```
