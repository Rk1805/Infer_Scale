"""Artifact schemas, actual aggregate values, and missing-data handling."""

import csv
import io
import json

import pytest

from inferscale.experiments.reporting import aggregate_csv, run_csv, write_reports
from inferscale.schemas.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRun


def fixture_result():
    return BenchmarkResult(
        config=BenchmarkConfig(policies=["fixed"], seeds=[11, 22]),
        runs=[
            BenchmarkRun(
                policy="fixed",
                workload="steady",
                seed=11,
                status="completed",
                config={"episode_duration": 30},
                metrics={"total_cost": 2, "p95_latency_ms": None},
            ),
            BenchmarkRun(
                policy="fixed",
                workload="steady",
                seed=22,
                status="completed",
                config={"episode_duration": 30},
                metrics={"total_cost": 4, "p95_latency_ms": None},
            ),
        ],
    )


def test_csv_schemas_and_values():
    result = fixture_result()
    rows = list(csv.DictReader(io.StringIO(run_csv(result))))
    assert len(rows) == 2
    assert {"policy", "workload", "seed", "status", "config_json", "total_cost"} <= rows[0].keys()
    assert json.loads(rows[0]["config_json"])["episode_duration"] == 30
    assert rows[0]["p95_latency_ms"] == ""
    summary = list(csv.DictReader(io.StringIO(aggregate_csv(result))))
    cost = next(row for row in summary if row["metric"] == "total_cost")
    assert float(cost["mean"]) == 3 and int(cost["n"]) == 2
    latency = next(row for row in summary if row["metric"] == "p95_latency_ms")
    assert latency["mean"] == "" and latency["n"] == "0"


def test_report_creation_and_failure_disclosure(tmp_path):
    result = fixture_result()
    result.runs[1].status = "failed"
    result.runs[1].error = "injected | failure"
    directory = write_reports(result, tmp_path, "benchmark-1")
    assert {p.name for p in directory.iterdir()} == {
        "runs.csv",
        "aggregates.csv",
        "report.md",
        "manifest.json",
    }
    report = (directory / "report.md").read_text(encoding="utf-8")
    assert "Successful runs: 1; failed runs: 1" in report
    assert "Student's t" in report and "N/A" in report and "injected" in report
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["result"]["runs"][0]["metrics"]["total_cost"] == 2
    with pytest.raises(FileExistsError):
        write_reports(result, tmp_path, "benchmark-1")


@pytest.mark.parametrize("identifier", ["../escape", "a/b", "", "C:\\elsewhere"])
def test_report_path_containment(tmp_path, identifier):
    with pytest.raises(ValueError):
        write_reports(fixture_result(), tmp_path, identifier)
