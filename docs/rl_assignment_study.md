# Reproducing the trained-agent study

The earlier report figure was a six-run, 20-second integration smoke test. All
three baselines achieved 72% on-time completion, and no PPO checkpoint was
evaluated. It was useful for checking accounting but not for assessing learned
autoscaling. The assignment report now uses the protocol below.

## Simulator corrections

- Holding capacity or rejecting an action no longer restarts the engine's
  cooldown. Only accepted scaling changes start a new cooldown. Previously,
  repeated holds could prevent a later capacity change indefinitely.
- Decode begins immediately after prefill. Completion applies decode time once;
  an isolated request now matches the analytical service model's total time.
- The observation Box permits unbounded nonnegative raw queue/work/latency
  features. The policy checkpoint stores a fixed 12-feature scaling layer.

Regression tests cover holds between scale events, rejected scale-down at the
minimum, exact phase durations, and large valid observations. Checkpoint tests
verify that the ordinary `PPOPolicy` loader preserves the feature scaling and
returns the same action after save/load.

## Workloads and control settings

Each episode lasts 180 simulated seconds. Traffic begins after a 10-second
warmup, runs through four 40-second phases, and ends with a 10-second drain.
Arrival gaps are exponentially distributed. Each phase's request rate has an
independent seeded multiplier between 0.8 and 1.2. Prompt tokens are uniform
integers from 100 to 600; output tokens are from 50 to 350.

| Profile | Nominal request rates in the four phases (requests/s) | Usage |
|---|---|---|
| Bursty | 2, 12, 3, 20 | Training, validation, test |
| Steady | 8, 8, 8, 8 | Training, validation, test |
| Shifted | 15, 2, 22, 4 | Test only |

Common settings: two initial replicas, one to eight allowed replicas,
concurrency two per replica, 5-second startup, 10-second cooldown, 5-second
decisions, 2-second response deadline, 95% SLO, and $0.0025 per replica-second.
Pricing is illustrative, not a cloud quote. These settings are explicit study
settings, not the original CLI training defaults.

## Split and checkpoint selection

- Training cycles seeds 100–199, alternating bursty and steady profiles.
- Validation uses seeds 501–503 on both training profile families.
- Reserved testing uses seeds 1001–1005 on all three profiles.
- No test trace is used for fitting input scales, learning, or checkpoint choice.
- One PPO random seed (7) is used. Workload seeds are not independent training
  replications, and uncertainty must not be described as training robustness.

The study compares entropy coefficients 0 and 0.02 on validation. Each candidate
uses 32,768 environment steps, learning rate 0.0003, discount 0.99, rollout size
512, minibatch 64, ten epochs, and two 128-unit hidden layers. Other reward and
environment settings are shared. The zero-step policy is recorded as an
untrained control but is ineligible for the trained checkpoint.

The checkpoint with greatest mean deterministic validation return among trained
checkpoints is eligible for selection. Ties prefer the earlier candidate/step.
The selected checkpoint is frozen before the reserved test evaluation. The
selection record and both candidates' validation evidence are retained.

## Commands

From the repository root, activate the environment and install dependencies:

```bash
python -m pip install -e ".[test,dev]"
python -m inferscale.experiments.rl_study train --directory artifacts/repro_default --entropy-coef 0
python -m inferscale.experiments.rl_study train --directory artifacts/repro_entropy --entropy-coef 0.02
python -m inferscale.experiments.rl_study select --directory artifacts/repro_selected --candidates artifacts/repro_default artifacts/repro_entropy
python -m inferscale.experiments.rl_study evaluate --directory artifacts/repro_selected
```

The released selected checkpoint and exact configuration are recorded in
`artifacts/rl_study/training_manifest.json`. Evaluate it without retraining:

```bash
python -m inferscale.experiments.rl_study evaluate
python deliverables/InferScale_Report/generate_assignment_plot.py
```

For a fresh single-candidate reproduction, pass a new `--directory` to both
training and evaluation. Training refuses to overwrite a completed manifest.
Do not select or retune using test outcomes.

## Reading the outputs

Every policy receives the same request records for each profile/seed. Raw traces,
trace hashes, per-run metrics, and representative timelines are preserved.
All four policies (fixed target four, custom threshold, queue, and PPO) use the
same engine, constraints, startup, tariff, horizon, and deadline.

The chart compares cost, confirmed on-time attainment, and each run's p95 latency.
Error bars are 95% Student t intervals over the five test traces, conditional on
one selected trained agent. Pending/overdue requests remain in SLO accounting;
latency percentiles alone describe completed requests. The learning panel uses
actual stochastic training returns and deterministic validation returns, which
are different measurements. Lower cost is not automatically better service.

These are analytical CPU simulations. They do not establish real GPU speedups,
production-trace generalisation, Kubernetes HPA equivalence, or PPO superiority.
