"""Fair baseline savings, conservative SLO decisions, and portable report contents."""

import json

import pytest

from inferscale.experiments.evaluation import cost_slo_comparisons
from inferscale.experiments.reporting import write_reports
from inferscale.schemas.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRun
from inferscale.schemas.experiment import ExperimentConfig


def result_fixture():
    config = BenchmarkConfig(experiment=ExperimentConfig(slo_target=0.95),
                             policies=["fixed", "queue"], seeds=[11, 22])
    runs = []
    for seed in config.seeds:
        for policy, cost in [("fixed", 10), ("queue", 6)]:
            runs.append(BenchmarkRun(
                policy=policy, workload="steady", seed=seed, status="completed",
                trace_hash=str(seed), config={"duration": 10, "policy": policy},
                metrics={"accounting_version": 2, "total_cost": cost,
                         "slo_attainment_lower_bound": 0.96, "slo_attainment_upper_bound": 0.98,
                         "pending_requests": 2, "overdue_requests": 0},
            ))
    return BenchmarkResult(config=config, runs=runs)


def test_matched_savings_and_slo_success():
    rows = cost_slo_comparisons(result_fixture())
    assert rows[0]["mean_cost_savings_percent"] == 0
    assert rows[1]["mean_cost_savings_percent"] == 40
    assert rows[1]["matched_cost_pairs"] == 2
    assert rows[1]["slo_status"] == "met"


def test_savings_are_mean_of_pairs_not_ratio_of_means():
    result = result_fixture()
    result.runs[2].metrics["total_cost"] = 20
    result.runs[3].metrics["total_cost"] = 18
    row = cost_slo_comparisons(result)[1]
    assert row["mean_cost_savings_percent"] == 25  # mean of 40% and 10%


def test_different_trace_or_settings_and_zero_cost_cannot_form_savings_pairs():
    result = result_fixture()
    result.runs[1].trace_hash = "different"
    result.runs[3].config["duration"] = 20
    assert cost_slo_comparisons(result)[1]["mean_cost_savings_percent"] is None
    result = result_fixture()
    for run in result.runs:
        if run.policy == "fixed":
            run.metrics["total_cost"] = 0
    assert cost_slo_comparisons(result)[1]["matched_cost_pairs"] == 0


def test_one_bad_seed_fails_even_when_mean_attainment_reaches_target():
    result = result_fixture()
    result.runs[1].metrics.update(slo_attainment_lower_bound=1, slo_attainment_upper_bound=1)
    result.runs[3].metrics.update(slo_attainment_lower_bound=0.9, slo_attainment_upper_bound=0.9)
    row = cost_slo_comparisons(result)[1]
    assert row["mean_attainment_lower"] == pytest.approx(0.95)
    assert row["slo_status"] == "violated"
    assert row["slo_met_runs"] == row["slo_violated_runs"] == 1


def test_pending_failed_and_unfinished_benchmarks_cannot_claim_slo_success():
    result = result_fixture()
    result.runs[3].metrics.update(slo_attainment_lower_bound=0.9, slo_attainment_upper_bound=1)
    assert cost_slo_comparisons(result)[1]["slo_status"] == "inconclusive"
    result.runs[3].status = "failed"
    row = cost_slo_comparisons(result)[1]
    assert row["slo_status"] == "inconclusive" and row["failed_runs"] == 1
    result.runs.pop()
    assert cost_slo_comparisons(result)[1]["slo_status"] == "inconclusive"


def test_legacy_results_are_not_presented_as_runtime_costs():
    result = result_fixture()
    for run in result.runs:
        run.metrics.pop("accounting_version")
    rows = cost_slo_comparisons(result)
    assert all(row["slo_status"] == "unavailable" for row in rows)
    assert all(row["mean_cost"] is None for row in rows)


def test_reports_include_cost_slo_and_version(tmp_path):
    directory = write_reports(result_fixture(), tmp_path, "cost-slo")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["cost_slo"][1]["mean_cost_savings_percent"] == 40
    report = (directory / "report.md").read_text(encoding="utf-8")
    assert "Cost and SLO evaluation" in report
    assert "Positive savings alone do not establish SLO compliance" in report
