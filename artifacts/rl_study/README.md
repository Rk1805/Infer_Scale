# InferScale trained PPO agent and evaluation

This directory contains a real trained Stable-Baselines3 PPO checkpoint and its
held-out synthetic evaluation. It is not a placeholder or a renamed baseline.

- Selected policy: `ppo_autoscaler.zip`, 32,768 observed environment steps,
  entropy coefficient 0.02, PPO training seed 7.
- Checkpoint SHA256:
  `5ba0eb959a62fb9466a79ed837c1172239b09bdfcef1b090aa67315c59f4378b`.
- Two 32,768-step candidates were compared using validation only. The untrained
  control was recorded but could not be selected. `selection.json` and
  `candidates/` retain that evidence.
- All 60 test runs use shared traces: four policies, three workload profiles,
  and five reserved request seeds. PPO inference is deterministic.
- PPO met the 95% SLO in 15/15 runs, fixed in 8/15, threshold in 5/15, and queue
  in 3/15. This does not establish a general policy winner.
- Bursty means: PPO 100% on time, $2.815, p95 0.690 s; fixed 77.30%, $1.775,
  p95 4.185 s. PPO protects deadlines at greater cost and overprovisions steady
  traffic. All figures are analytical CPU simulations.

Install the project (`python -m pip install -e ".[test,dev]"`) before loading
the checkpoint. Its fixed input scaling is implemented by
`inferscale.rl.features.ScaledObservations` and stored inside the model.
`requirements-study.txt` records the numeric-library versions used in this run.

From the repository root:

```bash
python -m inferscale.experiments.rl_study evaluate
python deliverables/InferScale_Report/generate_assignment_plot.py
```

Training, validation, exact test traces, per-run CSV/JSON, timelines, summaries,
and a fresh-process replay check are included. `verification.json` confirms
the saved model has optimizer updates and parameters different from its initial
policy, matching checkpoint/source hashes, all paired traces, no pending test
requests, and exactly reproduced fixed/PPO metrics for one test trace.

The evidence uses one training seed. Error bars reflect five request seeds for
the selected policy. Validation improved over early trained checkpoints but
remained below the untrained deterministic control. Neither GPU performance nor
production-trace generalisation is established. See root README and
`docs/rl_assignment_study.md` for configuration and fresh reproduction commands.
