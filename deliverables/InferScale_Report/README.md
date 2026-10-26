# InferScale project report

Authors: **Rishi Kaneria (B23CS1058)** and **Harsh Biyani (B23CS1019)**.

The PDF is exactly two pages in a two-column LaTeX format for the RL mini-project
assignment. Team names, roll numbers, and the clickable GitHub link appear at
the top. It defines the implemented 12-dimensional observation space, discrete
actions, mathematical reward and incentives, simulator, PPO configuration,
baseline comparisons, and measured example results. It has no separate cover or
bibliography. The study uses main revision `61e3b02` plus the simulator corrections
and training implementation recorded by hashes in `artifacts/rl_study/selection.json`.

- `InferScale_Project_Report.pdf`: finished report.
- `InferScale_Project_Report.tex`: editable source.
- `generate_assignment_plot.py`: four-panel chart from actual PPO/test evidence.
- `study_numbers.tex`: report values generated directly from the test results.
- `figures/assignment_results.pdf`: current measured results in vector format.

To rebuild with the project Python dependencies installed, run from the repository root:

```powershell
python deliverables/InferScale_Report/generate_assignment_plot.py
cd deliverables/InferScale_Report
tectonic InferScale_Project_Report.tex
```

The chart reads `artifacts/rl_study/` in the repository. These artifacts contain
the selected trained PPO checkpoint, both candidates' validation evidence,
training history, exact held-out traces, and all 60 comparison runs. The figure
shows cost, SLO attainment, run-level p95 latency, and actual learning history.
Install Tectonic separately to compile the LaTeX source. Results are synthetic
CPU simulation, not GPU or production-trace measurements. The original short
smoke-test material and generated distribution ZIPs are not part of this report.

The report meets the supplied two-page format and includes an actual trained
agent evaluation. Root README and `docs/rl_assignment_study.md` document setup,
model loading, splits, selection, and reproduction. Check private repository
access for the specified evaluators if applicable.
