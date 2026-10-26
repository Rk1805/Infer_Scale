# Benchmark verification record

This records the earlier advanced-benchmarking contribution. The subsequent runtime
accounting v2 and SLO changes are described in [cost_and_slo.md](cost_and_slo.md);
the statements below about unchanged metric formulas apply to the earlier work.

Validation ran locally on Windows with Python 3.13.2 and Node.js 24.14.1. No GPU
was required. Tests use isolated databases and temporary files; the existing
repository `experiments.db` was preserved byte-for-byte.

| Check | Observed result |
|---|---|
| Complete Python suite | 122 passed; four existing dependency deprecation warnings |
| Frontend ESLint | Passed, zero warnings |
| Frontend TypeScript and Vite production build | Passed; bundle size advisory remains |
| Live Uvicorn HTTP smoke test | Passed: six successful simulation runs, all four report downloads, and the existing experiments endpoint |
| Ruff on new Python modules/tests | Passed |
| Full-backend Ruff | 395 findings; original commit has 418; no newly introduced findings under a source-line comparison |
| Original commit's complete test suite | Timed out after 90 seconds |
| Isolated original simulator suite | Timed out in `test_episode_termination_with_pending_requests`; terminal queue handling and the test were repaired |
| Docker execution | Unverified: Docker is not installed in this environment |
| Interactive browser behavior | Unverified; frontend was checked through lint, type checking, and build |
| Empirical trained-PPO policy superiority | Not evaluated or claimed |

Executed commands:

```text
python -m pytest backend/inferscale/tests -q -p no:cacheprovider
npm run lint
npm run build
```

Coverage includes seeded trace equivalence at the actual simulator input, repeatable
results, configuration propagation, recurring decisions, zero arrivals, explicit
checkpoint failures, individual failed runs, known t-interval values, matching paired
differences, missing/non-finite observations, CSV/report contents, HTTP validation,
partial/terminal results, downloads, SQLite reopen/restart behavior, queue bounds,
contained checkpoint paths, and API responsiveness during background computation.

The actual PPO checkpoint compatibility test saves an **untrained test fixture** and
loads it for inference without training. It verifies integration, not learned policy
quality. The pre-existing RL tests separately perform short CPU training runs.

There are 49 new benchmark-focused tests and 73 existing tests. A second narrow
simulator repair clears terminal queued events when stepping after the horizon,
preventing callers that loop on the event queue from spinning indefinitely. It does
not change metric formulas or execute requests beyond the episode.

The frontend build repairs are limited to previously broken shared utility/client
imports, missing imports/types, select change callbacks, unused imports/variables,
and an invalid chart prop. No dashboard page was replaced. A package lock and ESLint
configuration make the existing install/build/lint scripts reproducible and usable.

Python package discovery now exposes `inferscale` from `backend/`. The API Dockerfile
copies package sources before installation and uses the standard-library HTTP client
for its health check. Container builds were not executed. The pre-existing optional
Compose worker still names an absent `inferscale.worker.main` module and does not
provide shared in-memory job coordination; the benchmark runs in the API process
with one API worker, as documented.
