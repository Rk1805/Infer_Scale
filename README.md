# InferScale

**Autonomous LLM Inference Autoscaling with Reinforcement Learning**

InferScale investigates whether reinforcement learning can reduce the compute cost of LLM inference serving while maintaining response-time service-level objectives (SLOs). The system compares four autoscaling policies:

1. **Fixed Capacity** - Always maintains a constant number of replicas
2. **Threshold Autoscaling** - Scales based on utilization/queue pressure thresholds
3. **Queue-Based Autoscaling** - Reacts to estimated waiting work per replica
4. **PPO (RL)** - Learns optimal scaling policy via Proximal Policy Optimization

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
cd InferScale

# Install Python dependencies
make install

# Install frontend dependencies
make frontend-install

# Generate synthetic data
make generate-data
```

### Development

```bash
# Start both API and frontend (two terminals)
make dev-api      # Terminal 1: API on http://localhost:8000
make dev-frontend # Terminal 2: Frontend on http://localhost:5173

# Or use Docker
make docker-up
```

### Training & Evaluation

```bash
# Short PPO training smoke test (5k timesteps)
make train-smoke

# Full training (200k timesteps)
cd backend && python -m inferscale.cli train --timesteps 200000

# Evaluate trained model
make evaluate

# Run policy comparison benchmark
make benchmark
```

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
- **Observation**: 12 normalized features (queue, arrival rate, replicas, latency, SLA rate, cooldown, etc.)
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
