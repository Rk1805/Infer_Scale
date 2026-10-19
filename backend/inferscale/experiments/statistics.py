"""Run-level Student's t summaries and matched policy differences."""

import json
import math
from collections import defaultdict
from itertools import combinations
from statistics import mean, stdev

from scipy.stats import t

from inferscale.schemas.benchmark import METRICS, BenchmarkResult


def summarize_metric(values, *, independent: bool = True) -> dict:
    """Summarize finite observations, using sample SD and a two-sided 95% t interval.

    Runs, not individual requests, are experimental replications. One sample has
    no sample SD or CI. Deterministic imported-trace repeats have no inferential CI.
    Bounds are intentionally not clipped to a metric's physical range.
    """
    clean = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]
    count = len(clean)
    result = dict(
        n=count,
        mean=None,
        std=None,
        minimum=None,
        maximum=None,
        ci_low=None,
        ci_high=None,
        ci_method="student_t_95" if independent else "unavailable",
    )
    if not clean:
        return result
    average = mean(clean)
    result.update(mean=average, minimum=min(clean), maximum=max(clean))
    if count >= 2:
        deviation = stdev(clean)
        result["std"] = deviation
        if independent:
            half_width = float(t.ppf(0.975, count - 1)) * deviation / math.sqrt(count)
            result.update(ci_low=average - half_width, ci_high=average + half_width)
    return result


def aggregate_results(result: BenchmarkResult) -> list[dict]:
    """Aggregate scalar metrics by workload/policy, retaining missing/failed counts."""
    groups = defaultdict(list)
    for run in result.runs:
        groups[run.workload, run.policy].append(run)
    metrics = sorted(set(METRICS) | {name for run in result.runs for name in run.metrics})
    rows = []
    for (workload, policy), runs in sorted(groups.items()):
        completed = [r for r in runs if r.status == "completed"]
        for metric in metrics:
            summary = summarize_metric(
                [r.metrics.get(metric) for r in completed],
                independent=not bool(result.config.trace_path),
            )
            rows.append(
                dict(
                    workload=workload,
                    policy=policy,
                    metric=metric,
                    attempted=len(runs),
                    failed=len(runs) - len(completed),
                    missing=len(completed) - summary["n"],
                    **summary,
                )
            )
    return rows


def paired_comparisons(result: BenchmarkResult) -> list[dict]:
    """Summarize A-minus-B differences only for matching seed, trace, and settings.

    No significance or policy-superiority claim is made. Each matched seed gives
    one difference. Failed or missing members are excluded with matched n exposed.
    """
    indexed = {}
    for run in result.runs:
        if run.status != "completed":
            continue
        common = {k: v for k, v in run.config.items() if k != "policy"}
        key = (run.workload, run.seed, run.trace_hash, json.dumps(common, sort_keys=True))
        # A malformed imported result must not multiply experimental replications.
        if (key, run.policy) in indexed:
            raise ValueError("Duplicate policy result for a matched experimental unit")
        indexed[key, run.policy] = run
    rows = []
    for workload in result.config.workloads:
        for policy_a, policy_b in combinations(result.config.policies, 2):
            differences = defaultdict(list)
            for (key, policy), run_a in indexed.items():
                if key[0] != workload or policy != policy_a:
                    continue
                run_b = indexed.get((key, policy_b))
                if run_b is None:
                    continue
                for metric, value_a in run_a.metrics.items():
                    value_b = run_b.metrics.get(metric)
                    if all(
                        isinstance(v, (int, float)) and math.isfinite(v) for v in (value_a, value_b)
                    ):
                        differences[metric].append(value_a - value_b)
            for metric, values in sorted(differences.items()):
                rows.append(
                    dict(
                        workload=workload,
                        policy_a=policy_a,
                        policy_b=policy_b,
                        metric=metric,
                        difference="policy_a_minus_policy_b",
                        **summarize_metric(values, independent=not bool(result.config.trace_path)),
                    )
                )
    return rows
