"""CSV, Markdown, and JSON artifacts backed exclusively by benchmark results."""

import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from inferscale.experiments.evaluation import cost_slo_comparisons
from inferscale.experiments.statistics import aggregate_results, paired_comparisons
from inferscale.schemas.benchmark import METRICS, BenchmarkResult

ARTIFACTS = {
    "runs": "runs.csv",
    "aggregates": "aggregates.csv",
    "report": "report.md",
    "manifest": "manifest.json",
}
SUMMARY_FIELDS = [
    "policy",
    "workload",
    "metric",
    "attempted",
    "failed",
    "missing",
    "n",
    "mean",
    "std",
    "minimum",
    "maximum",
    "ci_low",
    "ci_high",
    "ci_method",
]


def csv_text(rows: list[dict], fields: list[str]) -> str:
    """Serialize a stable CSV schema; unavailable values are empty cells."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def run_csv(result: BenchmarkResult) -> str:
    """Export one row for every attempted run, including failures and full settings."""
    metrics = sorted(set(METRICS) | {key for run in result.runs for key in run.metrics})
    fields = [
        "policy",
        "workload",
        "seed",
        "status",
        "error",
        "trace_hash",
        "request_count",
        "config_json",
        *metrics,
    ]
    rows = []
    for run in result.runs:
        rows.append(
            dict(
                policy=run.policy,
                workload=run.workload,
                seed=run.seed,
                status=run.status,
                error=run.error,
                trace_hash=run.trace_hash,
                request_count=run.request_count,
                config_json=json.dumps(run.config, sort_keys=True),
                **run.metrics,
            )
        )
    return csv_text(rows, fields)


def aggregate_csv(result: BenchmarkResult) -> str:
    """Export run-level summaries, units implied by metric names."""
    return csv_text(aggregate_results(result), SUMMARY_FIELDS)


def markdown_report(result: BenchmarkResult, benchmark_id: str) -> str:
    """Describe observed results, experimental uncertainty, and simulator limitations."""

    def cell(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    successful = sum(run.status == "completed" for run in result.runs)
    lines = [
        f"# InferScale Benchmark Report: {benchmark_id}",
        "",
        "## Configuration",
        "",
        f"- Policies: {', '.join(result.config.policies)}",
        f"- Workloads: {', '.join(result.config.workloads)}",
        f"- Seeds: {', '.join(map(str, result.config.seeds))}",
        f"- Duration: {result.config.experiment.episode_duration:g} simulation seconds",
        f"- Response deadline: {result.config.experiment.sla_threshold:g} seconds",
        f"- SLO target: {result.config.experiment.slo_target:.2%} of arrivals",
        f"- USD per billable replica-second: {result.config.experiment.cost_per_replica_second:g}",
        f"- Successful runs: {successful}; failed runs: {len(result.runs) - successful}",
        f"- Revision: {result.provenance.get('revision') or 'unavailable'}",
        f"- Working tree dirty: {result.provenance.get('working_tree_dirty')}",
        f"- Checkpoint SHA-256: {result.provenance.get('checkpoint_sha256', 'not evaluated')}",
        f"- Evaluation split: {cell(result.config.evaluation_split)}",
        "",
        "Full effective settings and trace hashes are in runs.csv and manifest.json.",
        "",
        "## Aggregate results",
        "",
        "Each observation is one experimental run. Percentiles are computed within each run; "
        "requests from different runs are never pooled.",
        "",
        "| Workload | Policy | Metric | Valid n | Mean | Sample SD | 95% CI |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in aggregate_results(result):
        interval = (
            f"[{cell(row['ci_low'])}, {cell(row['ci_high'])}]"
            if row["ci_low"] is not None
            else "N/A"
        )
        lines.append(
            "| "
            + " | ".join(cell(row[k]) for k in ("workload", "policy", "metric", "n", "mean", "std"))
            + f" | {interval} |"
        )
    lines += [
        "",
        "## Paired comparisons",
        "",
        "Differences are policy A minus policy B on matching workload, seed, trace hash, "
        "and simulator/baseline settings. Direction depends on the metric; no winner is assumed.",
        "",
        "| Workload | A | B | Metric | Matched n | Mean difference | 95% CI |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in paired_comparisons(result):
        interval = (
            f"[{cell(row['ci_low'])}, {cell(row['ci_high'])}]"
            if row["ci_low"] is not None
            else "N/A"
        )
        lines.append(
            "| "
            + " | ".join(
                cell(row[k]) for k in ("workload", "policy_a", "policy_b", "metric", "n", "mean")
            )
            + f" | {interval} |"
        )
    lines += [
        "",
        "## Cost and SLO evaluation",
        "",
        "Accounting v2 bills starting, active (including idle), and draining replica time. "
        "Legacy startup-only runs are excluded from this section. "
        "SLO status uses all arrivals: met requires the lower attainment bound to reach "
        "the target in every configured seed; violated means at least one run's upper "
        "bound is below target. Unfinished requests with unexpired deadlines are pending.",
        "",
        f"Baseline: {result.config.baseline_policy}. "
        "Savings use matched workload/seed/trace/settings "
        "pairs with nonzero baseline cost. Positive savings alone do not establish SLO compliance.",
        "",
        "| Workload | Policy | Mean cost USD | SLO status | Met / violated / inconclusive runs | "
        "Mean attainment bounds | Mean overdue / pending | Matched pairs | Mean savings % |",
        "|---|---|---:|---|---|---|---|---:|---:|",
    ]
    for row in cost_slo_comparisons(result):
        lines.append(
            f"| {row['workload']} | {row['policy']} | {cell(row['mean_cost'])} | "
            f"{row['slo_status']} | {row['slo_met_runs']} / {row['slo_violated_runs']} / "
            f"{row['slo_inconclusive_runs']} | "
            f"[{cell(row['mean_attainment_lower'])}, {cell(row['mean_attainment_upper'])}] | "
            f"{cell(row['mean_overdue_requests'])} / {cell(row['mean_pending_requests'])} | "
            f"{row['matched_cost_pairs']} | {cell(row['mean_cost_savings_percent'])} |"
        )
    lines += [
        "",
        "## Methodology and limitations",
        "",
        "95% confidence intervals use Student's t with n-1 degrees of freedom and sample "
        "standard deviation (ddof=1). They estimate a mean across independent workload runs, "
        "not a range containing 95% of requests or runs. With fewer than two independent "
        "runs, confidence bounds are unavailable. Bounds are not clipped. Small samples, "
        "Skewed metrics, failed-run selection, and multiple exploratory comparisons "
        "also limit inference.",
        "",
    ]
    lines.extend(f"- {warning}" for warning in result.warnings)
    lines += ["", "## Failed runs", ""]
    failures = [run for run in result.runs if run.status == "failed"]
    lines.extend(f"- {run.workload}/{run.seed}/{run.policy}: {cell(run.error)}" for run in failures)
    if not failures:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def write_reports(result: BenchmarkResult, reports_root: str | Path, benchmark_id: str) -> Path:
    """Create a new, contained report directory; existing artifacts are never overwritten."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", benchmark_id):
        raise ValueError("Invalid benchmark identifier")
    root = Path(reports_root).resolve()
    destination = root / benchmark_id
    if destination.resolve().parent != root:
        raise ValueError("Report directory escapes the configured root")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=2,
        benchmark_id=benchmark_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        result=result.model_dump(mode="json"),
        aggregates=aggregate_results(result),
        paired=paired_comparisons(result),
        cost_slo=cost_slo_comparisons(result),
    )
    contents = {
        "runs.csv": run_csv(result),
        "aggregates.csv": aggregate_csv(result),
        "report.md": markdown_report(result, benchmark_id),
        "manifest.json": json.dumps(manifest, indent=2, allow_nan=False),
    }
    for name, content in contents.items():
        temporary = destination / f"{name}.tmp"
        temporary.write_text(content, encoding="utf-8", newline="")
        temporary.replace(destination / name)
    return destination
