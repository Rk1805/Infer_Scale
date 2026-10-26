"""Cost comparisons with explicit arrival-based SLO bounds and matched baselines."""

import json
from statistics import mean

from inferscale.schemas.benchmark import BenchmarkResult


def cost_slo_comparisons(result: BenchmarkResult) -> list[dict]:
    """Describe each policy without treating censored or failed runs as SLO successes.

    Savings are means of paired percentage differences, not differences of unrelated
    aggregate costs. Old startup-only results cannot be mixed with accounting v2.
    """

    def key(run):
        settings = {k: v for k, v in run.config.items() if k != "policy"}
        return run.workload, run.seed, run.trace_hash, json.dumps(settings, sort_keys=True)

    def current(run):
        return run.status == "completed" and run.metrics.get("accounting_version") == 2

    baseline = {
        key(run): run for run in result.runs
        if run.policy == result.config.baseline_policy and current(run) and run.trace_hash
    }
    rows = []
    target = result.config.experiment.slo_target
    for workload in result.config.workloads:
        for policy in result.config.policies:
            runs = [r for r in result.runs if r.workload == workload and r.policy == policy]
            valid = [r for r in runs if current(r)]
            bounded = [r for r in valid if r.metrics.get("slo_attainment_lower_bound") is not None]
            passed = sum(r.metrics["slo_attainment_lower_bound"] >= target for r in bounded)
            violated = sum(r.metrics["slo_attainment_upper_bound"] < target for r in bounded)
            status = "inconclusive"
            if violated:
                status = "violated"
            elif passed == len(result.config.seeds):
                status = "met"
            elif not valid:
                status = "unavailable"

            savings = []
            for run in valid:
                reference = baseline.get(key(run))
                if reference and reference.metrics.get("total_cost", 0) > 0:
                    savings.append(
                        (reference.metrics["total_cost"] - run.metrics["total_cost"])
                        / reference.metrics["total_cost"] * 100
                    )

            def average(metric):
                values = [r.metrics[metric] for r in valid if r.metrics.get(metric) is not None]
                return mean(values) if values else None

            rows.append(dict(
                workload=workload,
                policy=policy,
                expected_runs=len(result.config.seeds),
                completed_runs=len(valid),
                failed_runs=sum(r.status == "failed" for r in runs),
                slo_status=status,
                slo_met_runs=passed,
                slo_violated_runs=violated,
                slo_inconclusive_runs=len(valid) - passed - violated,
                slo_target=target,
                mean_cost=average("total_cost"),
                mean_attainment_lower=average("slo_attainment_lower_bound"),
                mean_attainment_upper=average("slo_attainment_upper_bound"),
                mean_overdue_requests=average("overdue_requests"),
                mean_pending_requests=average("pending_requests"),
                baseline_policy=result.config.baseline_policy,
                matched_cost_pairs=len(savings),
                mean_cost_savings_percent=mean(savings) if savings else None,
            ))
    return rows
