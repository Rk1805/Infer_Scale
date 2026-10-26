# Runtime cost accounting and SLO evaluation

New simulations use accounting version 2. Saved experiments and benchmark artifacts
are not rewritten. Legacy startup-only results remain readable, but are excluded
from the new cost/SLO comparison; rerun them to compare operating costs.

## Billing rules

All replicas are billed continuously while **starting**, **active** (including idle),
or **draining**. Stopped replicas cost zero. The same configurable USD rate applies
to all three states; there is no additional flat startup fee or billing rounding.

`total_cost = cost_per_replica_second * total_replica_seconds`

The engine integrates state counts over every time interval before processing the
next event. Costs stop at the episode horizon. A startup longer than the episode
is charged only for the observed startup time. An idle replica stops immediately
on scale-down. A busy replica drains its in-flight requests and stops when its
last request completes; its draining time is billed.

For example, four replicas retained throughout a 600-second episode consume 2,400
billable replica-seconds. With 10-second startup time, that consists of 40 starting
and 2,360 active replica-seconds. At $0.01 per replica-second, cost is $24.00.

Results include the total and per-state replica-seconds, `starting_cost`,
`active_cost`, `draining_cost`, and `cost_per_on_time_request` (unavailable when no
request completes on time). `avg_active_replicas` is now a time average;
`avg_queue_depth` likewise integrates queue size over elapsed time.

This model is an explicit simulation tariff, not a cloud invoice. GPU/model
calibration, reservation discounts, network/storage charges, and billing quanta
are not modeled. The default rate comes from `INFERSCALE_GPU_COST_PER_REPLICA_SECOND`.

## Deadlines, pending requests, and SLO decisions

The response deadline is `arrival_time + sla_threshold`. Completing exactly at
the deadline is on time. Arrivals are observed in `[0, episode_duration)`;
completions exactly at the horizon are processed before final metrics.

Every observed arrival belongs to one of four disjoint groups:

| Outcome | Definition |
|---|---|
| On-time completion | Completed within its deadline |
| Late completion | Completed after its deadline |
| Overdue unfinished | Still queued or in flight after its deadline |
| Pending | Still unfinished, but its deadline has not passed |

An unfinished request exactly at its deadline is still pending. After that instant,
it is overdue. Pending requests are never silently counted as successes or failures.
Queued and in-flight counts are also reported separately.

`slo_violations = late_completions + overdue_requests`

`slo_evaluated_requests = completions + overdue_requests`

`sla_violation_rate = slo_violations / slo_evaluated_requests`

The last rate describes resolved outcomes only. For the SLO decision, all arrivals
are included through bounds on final attainment:

`lower = on_time_completions / arrivals`

`upper = (on_time_completions + pending_requests) / arrivals`

For target `slo_target` (default 0.99), the result is **met** if `lower >= target`,
**violated** if `upper < target`, and otherwise **inconclusive**. With no arrivals,
the bounds, violation rates, and SLO decision are unavailable (`null`). These bounds
describe unfinished outcomes; they are not statistical confidence intervals.

Example: among 100 arrivals, 90 finish on time, five finish late, three are overdue,
and two remain pending. Attainment is bounded by 90%–92%, so a 95% target is violated
regardless of how the pending requests eventually finish.

Latency percentiles and `completed_request_sla_violation_rate` still describe only
completed requests. Benchmark latency is unavailable when there are no completions.
The legacy `sla_violations` counter counts late completions only; use
`slo_violations` for late completions plus overdue unfinished requests.

## Benchmark comparison and exports

Open **Policy Comparison → Advanced Benchmarking**. Select policies, workload,
seeds, duration, response deadline, SLO target percentage, replica rate, and savings
baseline. The baseline must also be selected for execution to produce savings.

The cost/SLO table shows mean operating cost, attainment bounds, overdue/pending
counts, and the number of runs meeting or violating the target. A benchmark policy
is marked **met** only when every configured seed has a definite pass. One definite
violation marks it **violated**. Missing, failed, pending, or empty runs cannot turn
an otherwise inconclusive result into a pass. Policies with no current completed
runs are **unavailable**.

The scatter plot places mean cost against mean lower attainment bound. Its target
line is a visual reference; status is evaluated for each run, never from means alone.

Savings are calculated per matched workload, seed, trace hash, and effective settings:

`savings_percent = 100 * (baseline_cost - policy_cost) / baseline_cost`

The displayed mean averages those paired percentages. Failed, unmatched, legacy,
or zero-cost baseline runs do not contribute. The matched pair count is shown.
If the baseline is not evaluated, savings are unavailable. Savings are descriptive:
they do not establish SLO equivalence or policy superiority.

API results include `cost_slo`. CSV exports include state costs, replica-seconds,
and SLO counts/bounds; Markdown includes cost/SLO evaluation and baseline pair counts.
Manifest schema version 2 records the evaluation, accounting version, and billing
states alongside the existing configuration, code revision, and trace provenance.

```powershell
python -m inferscale.cli benchmark --policies fixed threshold queue --seeds 11 22 33 --workloads flash_crowd --duration 600 --sla-threshold 5 --slo-target 0.99 --cost-per-replica-second 0.01 --baseline-policy fixed
```

The same price and SLO fields are available in Experiment Studio and the experiment
API. Ordinary experiments normalize imported timestamps to zero and generate
synthetic traffic with a zero origin, so traffic falls within the simulation window.

## RL rewards

Every decision uses the cost incurred **during that interval**, normalized by the
cost of `max_replicas` over the same interval. Free pricing yields zero cost penalty.
New deadline violations are penalized once, even if a request later completes late.
The violation fraction divides new violations by previously pending plus newly
arrived requests. Queue pressure uses the current queue and active replica count;
churn counts actual accepted scaling actions in that interval.

`info.interval_metrics` exposes duration, cost, replica-seconds, new violations,
violation fraction, scaling actions, and the cost normalizer. `reward_components`
exposes the weighted cost, SLO, queue, and churn penalties. Older checkpoints remain
loadable, but were trained with different rewards; retrain before claiming benefits
from the corrected objective. Dashboard training remains simulated progress in the
inherited implementation; genuine PPO training is available through the CLI.

## Validation

`test_cost_slo.py` checks analytical state costs, partial intervals, startup beyond
the horizon, idle stop, busy drain, exact-deadline/horizon completions, all four
request outcomes, empty evidence, configuration validation, experiment propagation,
and incremental RL penalties. `test_cost_slo_evaluation.py` checks matched savings,
zero-cost baselines, legacy exclusion, pending/failed/partial runs, misleading
aggregate means, and exported reports. API lifecycle tests verify persisted fields
and configured tariff/target propagation. The existing simulator, policy, RL,
benchmark, statistics, and reporting suites provide regression coverage.

Local verification on Windows: **154 Python tests passed**, frontend production
build and ESLint passed, and new benchmark modules/tests passed Ruff. A live
six-run benchmark completed with all four report downloads. A headless Edge check
confirmed the saved benchmark, cost/SLO table, and scatter plot rendered. These
checks validate the implementation; they do not establish learned-policy superiority.
