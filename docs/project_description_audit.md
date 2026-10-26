# Project description audit

Audit date: 2026-10-27.

## Description being assessed

> InferScale: Autoscaling LLM inference servers with RL. A PPO agent observes request
> queue length, prompt/output token sizes, GPU utilisation and latency; actions scale
> GPU replicas up/down or adjust batch size; reward trades GPU cost against
> latency-SLA violations. Trained on real Azure/BurstGPT production traces versus
> Kubernetes threshold autoscaling.

## Verdict

The current implementation **partially matches** this description. It is a CPU-based
simulation and policy-benchmarking research prototype. Genuine PPO training exists
in the CLI, and the simulator supports replica scaling with runtime cost accounting
and arrival-based SLO evaluation. It does not yet establish a real GPU autoscaling
deployment, inference batch-size control, training on verified production traces,
or an empirical comparison against Kubernetes HPA.

Passing software tests establishes behavior covered by those tests. It does not
establish PPO superiority, production-trace training, or hardware-model fidelity.

## Requirement-by-requirement comparison

| Description requirement | Current implementation | Assessment |
|---|---|---|
| PPO agent | Stable-Baselines3 PPO, a Gymnasium environment, checkpoint saving/loading, and short real-training integration tests exist. CLI training calls `model.learn`. | Implemented for CPU simulation. |
| Request queue length | The 12-feature observation includes current waiting queue length and estimated queued work. | Implemented in simulation. |
| Prompt/output token sizes | Observations include mean token sizes from recently completed simulated requests. Canonical requests carry prompt/output token counts. | Implemented as historical simulated features, not live inference-server telemetry. |
| GPU utilisation | `SimulatorEngine.get_observation()` computes `active / max_replicas`. | Not actual GPU utilisation or busy-slot occupancy. |
| Latency observations | Observations include rolling p95 latency over completed simulated requests. | Implemented in simulation; not measured GPU serving latency. |
| Scale GPU replicas up/down | Actions modify simulated replicas through starting, active, draining, and stopped states. | Replica scaling is implemented in the simulator; no live GPU/Kubernetes controller is present. |
| Adjust inference batch size | The action space is `Discrete(3)`: down, hold, up. | Not implemented. PPO's `batch_size` controls optimizer minibatches, not inference batching. |
| GPU cost versus latency-SLA reward | Interval cost, new deadline violations, queue pressure, and accepted scaling actions contribute to the reward. Accounting v2 bills starting, idle/active, and draining replica time. | Implemented with an analytical simulator and configurable tariff, not measured hardware cost. |
| Azure/BurstGPT data support | Named CSV adapters and canonical CSV/JSON loading exist. | Partial: adapters expect specific column names; compatibility with actual upstream releases and a complete ingestion workflow are not established by the current tests. |
| Trained on real Azure/BurstGPT production traces | CLI training and evaluation generate synthetic workloads. No verified production-trace training run, split manifest, and associated checkpoint are included in the tracked repository. | Not demonstrated by the current implementation. |
| Versus Kubernetes threshold autoscaling | Fixed, custom threshold, and queue baselines exist. The threshold baseline uses upper/lower thresholds, cooldowns, and queue pressure. | A custom threshold comparison exists; an actual or faithfully reproduced Kubernetes HPA comparison is absent. |
| Research evaluation and reporting | Shared workload traces, multiple seeds, matched comparisons, confidence intervals, persisted jobs, exports, and cost/SLO outcomes are available. | Implemented for simulated policy evaluation; findings still require appropriate experiment design. |

Kubernetes HPA calculates desired replica counts using the relationship between
current and target metrics and applies controller behavior such as stabilization.
The repository's one-step threshold policy should not be described as Kubernetes
HPA without implementing and validating that equivalence. See the
[official HPA algorithm documentation](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/#algorithm-details).

## Source evidence

- [RL environment and three-action space](../backend/inferscale/rl/env.py):
  `InferScaleEnv.__init__`, `step`, and `_get_observation`.
- [Simulator and observation definitions](../backend/inferscale/simulator/engine.py):
  `get_observation`, `_handle_scale_action`, and `get_metrics`.
- [CLI training/evaluation](../backend/inferscale/cli.py): `train_ppo` and
  `evaluate_model` generate synthetic workloads; `benchmark --trace` accepts a
  trusted local canonical trace for evaluation.
- [PPO checkpoint behavior](../backend/inferscale/policies/ppo.py): the legacy
  policy falls back to threshold-like behavior without a model. The
  [advanced benchmark runner](../backend/inferscale/experiments/benchmark.py)
  explicitly rejects that fallback for PPO comparisons.
- [Reward components](../backend/inferscale/rl/reward.py) and
  [cost/SLO accounting rules](cost_and_slo.md).
- [Azure/BurstGPT adapters](../backend/inferscale/data/adapters/azure_burstgpt.py)
  and [canonical file loader](../backend/inferscale/data/loader.py).
- [Custom threshold policy](../backend/inferscale/policies/threshold.py).
- [API training, evaluation, and upload validation](../backend/inferscale/api/main.py):
  `run_training_background` generates simulated rewards and registers a model path;
  it does not perform PPO training or create the checkpoint. `/api/evaluations`
  returns a placeholder job, and dataset upload validation is also a placeholder.

## Remaining work to substantiate the supplied description

1. Add verified Azure/BurstGPT ingestion with schema tests, trace hashes, explicit
   train/validation/test partitions, and a training path that accepts those traces.
2. Connect dashboard training and evaluation to genuine background PPO jobs,
   actual checkpoint artifacts, real progress callbacks, and held-out evaluation.
3. Replace the utilisation proxy with a defensible simulated occupancy metric or
   measured GPU telemetry; calibrate latency/service capacity against a serving engine.
4. Add inference batch-size decisions, batch scheduling/latency semantics, and
   tests. Preserve a replica-only policy comparison for ablations.
5. Implement and validate an HPA-equivalent baseline or run a real Kubernetes HPA
   deployment, using the same workload and comparable SLO/cost definitions.
6. Run and retain reproducible trained-model evaluations against the baselines,
   including failures and uncertainty. Establish performance claims from results.

These are separate engineering/research tasks. This audit does not silently claim
that merging the existing branches implements them.

## Validation and deployment boundary

The delivery is checked with the complete Python test suite, the TypeScript/Vite
production build, frontend ESLint, and local browser checks across all seven pages,
including desktop/mobile layouts, theme persistence, and navigation. The merged
`main` branch is checked again before it is pushed.

Existing Python dependency deprecation warnings and Vite's bundle-size advisory
do not represent test failures. The repository also has inherited backend lint/type
debt; a passing test suite does not mean every static-analysis tool is clean.

No real GPU benchmark, upstream production-trace training study, or Kubernetes HPA
deployment was performed for this delivery. Docker execution is unverified in this
environment. The optional Compose worker references an absent
`inferscale.worker.main` module, so the documented optional worker deployment also
needs repair before it can be claimed operational. API authentication and production
operational hardening are separate from the tested local research workflow.

## Accurate description of the current project

InferScale is a CPU-based research simulator and dashboard for studying LLM
inference autoscaling. It provides a Gymnasium environment and PPO training CLI,
simulated replica-scaling actions, token/queue/latency observations, and a reward
that trades simulated compute cost against deadline violations. It compares
policies with fixed, custom threshold, and queue-based baselines using reproducible
benchmarks, statistical summaries, and cost/SLO reports. Canonical trace loading
and preliminary Azure/BurstGPT adapters are available; real production-trace
training, GPU telemetry, inference batch-size control, and Kubernetes HPA validation
remain future work.
