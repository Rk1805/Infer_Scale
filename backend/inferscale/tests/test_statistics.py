"""Numerical checks against known values, not copies of the implementation."""

import pytest

from inferscale.experiments.statistics import (
    aggregate_results,
    paired_comparisons,
    summarize_metric,
)
from inferscale.schemas.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRun


def test_known_student_t_summary():
    summary = summarize_metric([1, 2, 3, 4, 5])
    assert summary["n"] == 5
    assert summary["mean"] == 3
    assert summary["std"] == pytest.approx(1.58113883008419)
    assert summary["minimum"] == 1 and summary["maximum"] == 5
    assert summary["ci_low"] == pytest.approx(1.0367568385)
    assert summary["ci_high"] == pytest.approx(4.9632431615)


def test_empty_single_and_nonfinite():
    assert summarize_metric([])["n"] == 0
    single = summarize_metric([4, None, float("inf"), float("nan"), "missing"])
    assert single["n"] == 1 and single["mean"] == 4
    assert single["std"] is None and single["ci_low"] is None
    assert summarize_metric([3, 3, 3])["ci_low"] == 3
    assert summarize_metric([1, 2], independent=False)["ci_low"] is None


def result_fixture():
    runs = []
    for seed, a, b in [(11, 10, 7), (22, 20, 15), (33, 30, 23)]:
        for policy, value in [("fixed", a), ("queue", b)]:
            runs.append(
                BenchmarkRun(
                    policy=policy,
                    workload="steady",
                    seed=seed,
                    status="completed",
                    trace_hash=str(seed),
                    config={"policy": policy, "duration": 20},
                    metrics={"p95_latency_ms": value},
                )
            )
    return BenchmarkResult(
        config=BenchmarkConfig(policies=["fixed", "queue"], seeds=[11, 22, 33]), runs=runs
    )


def test_paired_differences_and_matching():
    result = result_fixture()
    summary = paired_comparisons(result)[0]
    assert summary["n"] == 3 and summary["mean"] == 5 and summary["std"] == 2
    result.runs[1].trace_hash = "different"
    assert paired_comparisons(result)[0]["n"] == 2
    result.runs[3].status = "failed"
    assert paired_comparisons(result)[0]["n"] == 1
    assert paired_comparisons(result)[0]["ci_low"] is None


def test_failed_and_missing_are_excluded():
    result = result_fixture()
    result.runs[0].status = "failed"
    result.runs[2].metrics["p95_latency_ms"] = None
    row = next(
        r
        for r in aggregate_results(result)
        if r["policy"] == "fixed" and r["metric"] == "p95_latency_ms"
    )
    assert row["n"] == 1 and row["failed"] == 1 and row["missing"] == 1
    assert row["mean"] == 30


def test_imported_trace_repeats_have_no_inferential_ci():
    result = result_fixture()
    result.config.trace_path = "trace.csv"
    assert all(row["ci_low"] is None for row in aggregate_results(result))
    assert all(row["ci_low"] is None for row in paired_comparisons(result))


def test_paired_differences_reject_different_configuration():
    result = result_fixture()
    result.runs[1].config["duration"] = 30
    assert paired_comparisons(result)[0]["n"] == 2
