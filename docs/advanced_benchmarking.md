# Advanced Benchmarking, Statistical Evaluation & Reporting

This contribution adds a reproducible research evaluation layer to InferScale. It now
includes [runtime cost accounting and arrival-based SLO evaluation](cost_and_slo.md).
The
original CLI compared each policy once per workload; the dashboard launched separate
experiments. Single-experiment CSV/Markdown exports already existed. Neither workflow
provided independent workload seeds, confidence intervals, matched comparisons, or
a persisted multi-run benchmark with trace and checkpoint provenance.

## Architecture and data flow

1. `schemas/benchmark.py` validates a `BenchmarkConfig` containing an existing
   `ExperimentConfig`, policy settings, workloads, seeds, and optional checkpoint.
2. `experiments/benchmark.py::prepare_workload` invokes the existing synthetic
   generators with a local `random.Random(seed)` and an explicit zero time origin.
   Canonical imported CSV/JSON traces use the existing loader and are shifted to zero.
3. Each workload/seed trace is sorted, clipped to the trace window and episode, and
   hashed. Every policy receives a deep copy of that same trace and a fresh existing
   policy instance. Policies receive current observations, never future requests.
4. The existing `SimulatorEngine` processes events and the existing policies select
   actions. A focused engine fix schedules recurring decision events; the previous
   direct-evaluation loop only received the decision at time zero. This does not
   replace the simulator, service model, policies, or PPO training pipeline.
5. `extract_metrics` produces scalar run observations. Individual failures are
   retained alongside successful runs; no policy is substituted for a failed policy.
6. `experiments/statistics.py` aggregates independent run values and matched
   differences. `experiments/reporting.py` exports the actual results.
7. `services/benchmark.py` runs a bounded thread executor and persists progress/results
   through the existing `ExperimentDB`. `api/benchmarks.py` exposes these records.
8. `BenchmarkPanel.tsx` extends Policy Comparison using Axios, existing UI controls,
   polling hooks, and Recharts with confidence error bars. Saved IDs are shareable
   through the `?benchmark=<id>` query parameter.

The Python configuration, simulator, and baseline policy parameters are shared
across policies. `replica_startup_delay` and `scaling_cooldown` are explicitly mapped
to the simulator/policy parameter names `startup_delay` and `cooldown`.

## Installation and local usage

Use Python 3.11+ in an activated virtual environment. No GPU or external workload
download is required. Install the existing dependencies:

```powershell
python -m pip install -e ".[test,dev]"
python -m inferscale.cli benchmark --policies fixed threshold queue --seeds 11 22 33 44 55 --workloads steady flash_crowd --duration 600 --num-requests 5000 --output results/benchmark.json
```

The existing `benchmark`, `--output`, `--include-ppo`, and `--ppo-model` options are
retained. JSON output now contains structured `config`, `runs`, `provenance`, and
`warnings`, rather than the old single-run workload/policy mapping. Default
benchmarking evaluates five seeds. The CLI exits nonzero if any run fails, after
preserving results and writing reports.

Python callers can avoid the API and database entirely at the runner interface:

```python
from inferscale.schemas.benchmark import BenchmarkConfig
from inferscale.schemas.experiment import ExperimentConfig
from inferscale.experiments.benchmark import run_benchmark
from inferscale.experiments.reporting import write_reports

config = BenchmarkConfig(
    experiment=ExperimentConfig(episode_duration=600, decision_interval=5),
    policies=["fixed", "threshold", "queue"],
    workloads=["flash_crowd"], seeds=[11, 22, 33, 44, 55],
)
result = run_benchmark(config)
write_reports(result, "reports/benchmarks", "assessment-demo")
```

The package retains its pre-existing eager imports, including global SQLite
initialization. Set `INFERSCALE_DB_URL` to an isolated database for a standalone
demonstration if the repository database must remain untouched.

### PPO and evaluation separation

```powershell
python -m inferscale.cli benchmark --include-ppo --ppo-model models/ppo_full.zip --seeds 101 202 303 --workloads flash_crowd --duration 600
```

PPO uses the existing `PPOPolicy` and deterministic prediction, loaded explicitly on
CPU. Benchmarks never train. Missing, corrupt, or incompatible checkpoints produce
explicit failed runs. The legacy non-benchmark policy fallback remains unchanged.
API training currently records simulated progress and model metadata; a registered
model is not proof that a checkpoint exists. The benchmark API checks the file.

Use independent evaluation traces/seeds rather than reusing a training trace.
`--evaluation-split` records a caller-supplied split label, not a verified guarantee.
The original training metadata does not establish trace hashes or a reproducible
training split; reports default to `unverified` and make no held-out performance claim.

### Imported traces

`--trace path/to/canonical.csv --workloads steady` loads the existing canonical
`RequestSchema` columns: `id`, `arrival_time_seconds`, `prompt_tokens`, `output_tokens`,
and optional `source`. IDs must be unique. The workload name is a grouping label for
an imported trace. Existing source adapters can prepare canonical Azure/BurstGPT data.

The simulator is deterministic for a fixed trace. Merely changing engine seeds does
not create independent observations: imported-trace repeats therefore have descriptive
summaries but no inferential confidence interval. HTTP clients cannot submit file paths.

## Metrics and statistical interpretation

Each independent workload seed contributes **one value per metric per policy**.
For example, a p95 value is computed within each run using the existing simulator
percentile definition; the statistical layer then summarizes those run-level p95s.
Requests across runs are never pooled as experimental replications.

Summaries contain valid `n`, mean, sample standard deviation (`ddof=1`), minimum,
maximum, attempted/failed/missing counts, and a two-sided 95% Student's t interval:

`mean ± t(0.975, n - 1) × sample_sd / sqrt(n)`

Empty/all-missing input has `n=0` and unavailable statistics. One valid run has a mean
but no sample standard deviation or interval. Non-finite/missing observations and
failed runs are excluded from that metric's sample count and disclosed. Bounds are
not clipped to physical ranges. Small samples and skewness may limit t approximations.
Intervals estimate uncertainty in the **mean**, not 95% of individual requests/runs.

Paired comparisons summarize `policy_a - policy_b` only for matching workload, seed,
trace hash, and effective settings. Failed or missing members cannot form a pair.
No significance test, multiple-comparison-adjusted conclusion, or automatic winner
is asserted. Failed-run selection can bias the remaining successful sample.

| Metric | Definition and limitation |
|---|---|
| `p50/p95/p99_latency_ms` | Existing run-level percentile definitions over completed requests. Missing when none complete. |
| `mean_latency_ms` | Mean of recorded completion latencies, converted from seconds to milliseconds. |
| `completed_request_sla_violation_rate` | Recorded violations / completions; unavailable with no completions. |
| `throughput`, `goodput` | Completions, or SLA-compliant completions, divided by configured episode seconds. Zero with no completions. |
| `unfinished_requests` | Recorded arrivals minus completions at the horizon. Consider alongside completion-only latency/SLA metrics. |
| `total_cost` | Accounting v2: billable starting + active + draining replica-seconds times the configured rate. Includes idle active time. This is a simulation estimate. |
| `sla_violation_rate` | Late completions plus overdue unfinished requests, divided by completed plus overdue requests. Unavailable with no resolved requests. |
| `slo_attainment_lower_bound`, `slo_attainment_upper_bound` | On-time completions / arrivals, and (on-time + pending) / arrivals. Unavailable with no arrivals. These are censoring bounds, not confidence intervals. |
| `max_queue_depth`, `scaling_events` | Existing recorded counters. |

`avg_active_replicas` is now the integral of active replica count divided by elapsed
time. Observation `utilization` still describes active/max replicas rather than
busy-slot occupancy. New results carry `accounting_version=2`; existing stored results
are preserved and must be rerun for cost/SLO comparisons.
An empty workload has zero arrivals/throughput and unavailable latency/SLA values.

## Reports and reproducibility

Artifacts live in `reports/benchmarks/<benchmark_id>/`:

- `runs.csv`: every attempted run, seed, status/error, trace hash, request count,
  full effective configuration as JSON, and scalar metrics.
- `aggregates.csv`: policy/workload/metric summaries, counts, sample SD, and CI bounds.
- `report.md`: configuration, observed comparisons, paired differences, failures,
  confidence methodology, simulator limitations, and interpretation.
- `manifest.json`: schema version, complete results, summaries, paired comparisons,
  code revision/dirty state, evaluation label, and evaluated checkpoint SHA-256.

Missing CSV values are empty; Markdown uses `N/A`. Existing artifact directories
are never overwritten. Generated results/reports are Git-ignored. A dirty revision
is not a complete source snapshot: commit code before producing final assessment
results and retain the full configuration/checkpoint separately.

## FastAPI and dashboard

Start the existing API with `python -m uvicorn inferscale.api.main:app --port 8000`.

```text
POST /api/benchmarks                         # 202: queued job
GET  /api/benchmarks                         # recent jobs
GET  /api/benchmarks/{uuid}                   # state and progress
GET  /api/benchmarks/{uuid}/results           # raw runs, aggregates, paired results
GET  /api/benchmarks/{uuid}/export?artifact=runs
```

Export artifacts: `runs`, `aggregates`, `report`, `manifest`. Unknown IDs return
404; malformed IDs/requests return 422; unavailable results or unfinished reports
return 409; a full queue returns 429. Reports use allowlisted names and contained
server paths. API PPO selection uses a registered `model_id` whose existing `.zip`
must resolve within the configured models directory. HTTP workloads are synthetic;
trusted local callers may load canonical traces.

Example request body:

```json
{"experiment":{"episode_duration":600},"policies":["fixed","threshold","queue"],"workloads":["flash_crowd"],"seeds":[11,22,33,44,55],"num_requests":5000}
```

Run the API with **one worker**, consistent with the existing in-process experiment
architecture. One benchmark executes at a time; at most eight may be outstanding.
Progress and partial results persist after each run. A benchmark with some failed
runs completes with explicit failure counts; all failed runs produce a failed job.
Orderly shutdown finishes queued work. On restart, abandoned queued/running records
are marked failed instead of pretending to resume. Existing SQLite tables and records
are preserved; only an additive `benchmarks` table is created.

In `frontend/`, run `npm ci` and `npm run dev`. Open **Policy Comparison → Advanced
Benchmarking**. Select policies, workload, seeds, duration, and optionally a registered
PPO model. The panel polls progress, displays run means and CIs with valid counts,
discloses errors, loads saved jobs, and downloads reports. Legacy comparison remains
available separately; its independently generated workloads are not the controlled
benchmark protocol.

## Tests and assessment demonstration

```powershell
python -m pytest backend/inferscale/tests/test_benchmark.py backend/inferscale/tests/test_statistics.py backend/inferscale/tests/test_reporting.py backend/inferscale/tests/test_benchmark_api.py -q
python -m pytest backend/inferscale/tests -q
cd frontend
npm run lint
npm run build
```

Benchmark tests use tiny CPU simulations, local fixtures, temporary databases, and
mock checkpoint loading. No GPU, network, or PPO training is required by the new tests.
Existing RL integration tests do perform short training runs. The shared test setup
isolates SQLite so tests cannot add records to the repository database.

For a viva: show identical per-seed trace hashes across policies, rerun with identical
configuration, inspect one failed PPO run, explain the distinction between request
percentiles and run-level confidence intervals, then download and trace report values
back to the corresponding `runs.csv` rows. Do not present simulation values as real
GPU performance or claim PPO is superior without measured supporting evidence.

See [the verification record](benchmark_verification.md) for executed checks,
pre-existing failures, deployment limits, and the scope of compatibility repairs.
